
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
```
mut_var_info_mtx(x::AbstractMatrix{<:Real}, bins_info::Union{Symbol, <:Integer} = :KN,
                 normalise::Bool = true)
```

Compute the mutual information and variation of information matrices.
"""
function mut_var_info_mtx(x::AbstractMatrix{<:Real},
                          bins_info::Union{Symbol, <:Integer} = :KN, normalise::Bool = true)
    calc_num_bins = if bins_info == :KN
        (xj, xi, j, i, T) -> _calc_num_bins(xj, xi, j, i,
                                            pyimport("astropy.stats").knuth_bin_width)

    elseif bins_info == :FD
        (xj, xi, j, i, T) -> _calc_num_bins(xj, xi, j, i,
                                            pyimport("astropy.stats").freedman_bin_width)

    elseif bins_info == :SC
        (xj, xi, j, i, T) -> _calc_num_bins(xj, xi, j, i,
                                            pyimport("astropy.stats").scott_bin_width)

    elseif bins_info == :HGR
        function (xj, xi, j, i, T)
            corr = cor(xj, xi)
            bins = corr == 1 ? _calc_num_bins(T) : _calc_num_bins(T, corr)
            return bins
        end
    else
        (xj, xi, j, i, T) -> bins_info
    end

    T, N = size(x)
    xtype = eltype(x)
    mut_mtx = Matrix{xtype}(undef, N, N)
    var_mtx = Matrix{xtype}(undef, N, N)

    for j ∈ 1:N
        xj = x[:, j]
        for i ∈ 1:j
            xi = x[:, i]
            bins = calc_num_bins(xj, xi, j, i, T)

            ex, ey, hxy = _calc_hist_data(xj, xi, bins)

            mut_ixy = mutualinfo(hxy)
            var_ixy = ex + ey - 2 * mut_ixy
            if normalise
                vxy = ex + ey - mut_ixy
                var_ixy = var_ixy / vxy
                mut_ixy /= min(ex, ey)
            end

            if abs(mut_ixy) < eps(typeof(mut_ixy)) || mut_ixy < zero(xtype)
                mut_ixy = zero(xtype)
            end
            if abs(var_ixy) < eps(typeof(var_ixy)) || var_ixy < zero(xtype)
                var_ixy = zero(xtype)
            end

            mut_mtx[i, j] = mut_ixy
            var_mtx[i, j] = var_ixy
        end
    end

    return Symmetric(mut_mtx, :U), Symmetric(var_mtx, :U)
end

function cordistance(v1::AbstractVector, v2::AbstractVector,
                     opt::DistcorOpt = DistcorOpt(;))
    N = length(v1)
    @smart_assert(N == length(v2) && N > 1)

    N2 = N^2

    method = opt.method
    args = opt.args

    a = pairwise(method, v1, args...)
    b = pairwise(method, v2, args...)
    A = a .- mean(a; dims = 1) .- mean(a; dims = 2) .+ mean(a)
    B = b .- mean(b; dims = 1) .- mean(b; dims = 2) .+ mean(b)

    dcov2_xx = sum(A .* A) / N2
    dcov2_xy = sum(A .* B) / N2
    dcov2_yy = sum(B .* B) / N2

    val = sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))

    return val
end

function cordistance(x::AbstractMatrix, opt::DistcorOpt = DistcorOpt(;))
    N = size(x, 2)

    mtx = Matrix{eltype(x)}(undef, N, N)
    for j ∈ 1:N
        xj = x[:, j]
        for i ∈ 1:j
            mtx[i, j] = cordistance(x[:, i], xj, opt)
        end
    end

    return Symmetric(mtx, :U)
end

function ltdi_mtx(x, alpha = 0.05)
    T, N = size(x)
    k = ceil(Int, T * alpha)
    mtx = Matrix{eltype(x)}(undef, N, N)

    @inbounds if k > 0
        for j ∈ 1:N
            xj = x[:, j]
            v = sort(xj)[k]
            maskj = xj .<= v
            for i ∈ 1:j
                xi = x[:, i]
                u = sort(xi)[k]
                ltd = sum(xi .<= u .&& maskj) / k
                mtx[i, j] = clamp(ltd, zero(eltype(x)), one(eltype(x)))
            end
        end
    end

    return Symmetric(mtx, :U)
end

function _gerber0_norm(x, mean_vec, std_vec, threshold)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)

    @inbounds for j ∈ 1:N
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = 0
            pos = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (x[k, i] - mui) / sigmai
                xj = (x[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += 1
                end
            end
            den = (pos + neg)
            mtx[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

function _gerber0(x, std_vec, threshold)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)

    @inbounds for j ∈ 1:N
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = 0
            pos = 0
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = x[k, i]
                xj = x[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += 1
                end
            end
            den = (pos + neg)
            mtx[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

"""
```
gerber0(x::AbstractMatrix, opt::GerberOpt = GerberOpt(;))
```
"""
function gerber0(x::AbstractMatrix, opt::GerberOpt = GerberOpt(;))
    threshold = opt.threshold
    normalise = opt.normalise

    std_func = opt.std_func.func
    std_args = opt.std_func.args
    std_kwargs = opt.std_func.kwargs
    std_vec = vec(std_func(x, std_args...; std_kwargs...))

    mtx = if normalise
        mean_func = opt.mean_func.func
        mean_args = opt.mean_func.args
        mean_kwargs = opt.mean_func.kwargs
        mean_vec = vec(mean_func(x, mean_args...; mean_kwargs...))
        _gerber0_norm(x, mean_vec, std_vec, threshold)
    else
        _gerber0(x, std_vec, threshold)
    end

    posdef_fix!(mtx, opt.posdef; msg = "Gerber0 Correlation ", cov_flag = false)

    return mtx, Symmetric(mtx .* (std_vec * transpose(std_vec)), :U)
end

function _gerber1_norm(x, mean_vec, std_vec, threshold)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)
    @inbounds for j ∈ 1:N
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = 0
            pos = 0
            nn = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (x[k, i] - mui) / sigmai
                xj = (x[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += 1
                end
            end
            den = (T - nn)
            mtx[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

function _gerber1(x, std_vec, threshold)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)
    @inbounds for j ∈ 1:N
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = 0
            pos = 0
            nn = 0
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = x[k, i]
                xj = x[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += 1
                end
            end
            den = (T - nn)
            mtx[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

"""
```
gerber1(x::AbstractMatrix, opt::GerberOpt = GerberOpt(;))
```
"""
function gerber1(x::AbstractMatrix, opt::GerberOpt = GerberOpt(;))
    threshold = opt.threshold
    normalise = opt.normalise

    std_func = opt.std_func.func
    std_args = opt.std_func.args
    std_kwargs = opt.std_func.kwargs
    std_vec = vec(std_func(x, std_args...; std_kwargs...))

    mtx = if normalise
        mean_func = opt.mean_func.func
        mean_args = opt.mean_func.args
        mean_kwargs = opt.mean_func.kwargs
        mean_vec = vec(mean_func(x, mean_args...; mean_kwargs...))
        _gerber1_norm(x, mean_vec, std_vec, threshold)
    else
        _gerber1(x, std_vec, threshold)
    end

    posdef_fix!(mtx, opt.posdef; msg = "Gerber1 Correlation ", cov_flag = false)

    return mtx, Symmetric(mtx .* (std_vec * transpose(std_vec)), :U)
end

function _gerber2_norm(x, mean_vec, std_vec, threshold)
    T, N = size(x)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)

    @inbounds for i ∈ 1:N
        xi = (x[:, i] .- mean_vec[i]) / std_vec[i]
        ti = threshold
        U[:, i] .= xi .>= ti
        D[:, i] .= xi .<= -ti
    end

    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc

    UmD = U - D
    H = transpose(UmD) * (UmD)

    h = sqrt.(diag(H))

    return H ./ (h * transpose(h))
end

function _gerber2(x, std_vec, threshold)
    T, N = size(x)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)

    @inbounds for i ∈ 1:N
        xi = x[:, i]
        ti = threshold * std_vec[i]
        U[:, i] .= xi .>= ti
        D[:, i] .= xi .<= -ti
    end

    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc

    UmD = U - D
    H = transpose(UmD) * (UmD)

    h = sqrt.(diag(H))

    return H ./ (h * transpose(h))
end

"""
```
gerber2(x::AbstractMatrix, opt::GerberOpt = GerberOpt(;))
```
"""
function gerber2(x::AbstractMatrix, opt::GerberOpt = GerberOpt(;))
    threshold = opt.threshold
    normalise = opt.normalise

    std_func = opt.std_func.func
    std_args = opt.std_func.args
    std_kwargs = opt.std_func.kwargs
    std_vec = vec(std_func(x, std_args...; std_kwargs...))

    mtx = if normalise
        mean_func = opt.mean_func.func
        mean_args = opt.mean_func.args
        mean_kwargs = opt.mean_func.kwargs
        mean_vec = vec(mean_func(x, mean_args...; mean_kwargs...))
        _gerber2_norm(x, mean_vec, std_vec, threshold)
    else
        _gerber2(x, std_vec, threshold)
    end

    posdef_fix!(mtx, opt.posdef; msg = "Gerber2 Correlation ", cov_flag = false)

    return mtx, Symmetric(mtx .* (std_vec * transpose(std_vec)), :U)
end

function _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
    # Zone of confusion.
    # If the return is not a significant proportion of the standard deviation, we classify it as noise.
    if abs(xi) < sigmai * c1 && abs(xj) < sigmaj * c1
        return zero(eltype(xi))
    end

    # Zone of indecision.
    # Center returns at mu = 0 and sigma = 1.
    ri = abs((xi - mui) / sigmai)
    rj = abs((xj - muj) / sigmaj)
    # If the return is less than c2 standard deviations, or greater than c3 standard deviations, we can't make a call since it may be noise, or overall market forces.
    if ri < c2 && rj < c2 || ri > c3 && rj > c3
        return zero(eltype(xi))
    end

    kappa = sqrt((1 + ri) * (1 + rj))
    gamma = abs(ri - rj)

    return kappa / (1 + gamma^n)
end

function _sb0_norm(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)
    @inbounds for j ∈ 1:N
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(x))
            pos = zero(eltype(x))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (x[k, i] - mui) / sigmai
                xj = (x[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, zero(eltype(x)), zero(eltype(x)),
                                     one(eltype(x)), one(eltype(x)), c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, zero(eltype(x)), zero(eltype(x)),
                                     one(eltype(x)), one(eltype(x)), c1, c2, c3, n)
                end
            end
            den = (pos + neg)
            mtx[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

function _sb0(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)

    @inbounds for j ∈ 1:N
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(x))
            pos = zero(eltype(x))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = x[k, i]
                xj = x[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                end
            end
            den = (pos + neg)
            mtx[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

"""
```
sb0(x::AbstractMatrix, gerberopt::GerberOpt = GerberOpt(;), sbopt::SBOpt = SBOpt(;))
```
"""
function sb0(x::AbstractMatrix, gerberopt::GerberOpt = GerberOpt(;),
             sbopt::SBOpt = SBOpt(;))
    c1 = sbopt.c1
    c2 = sbopt.c2
    c3 = sbopt.c3
    n = sbopt.n

    threshold = gerberopt.threshold
    normalise = gerberopt.normalise

    mean_func = gerberopt.mean_func.func
    mean_args = gerberopt.mean_func.args
    mean_kwargs = gerberopt.mean_func.kwargs
    mean_vec = vec(mean_func(x, mean_args...; mean_kwargs...))

    std_func = gerberopt.std_func.func
    std_args = gerberopt.std_func.args
    std_kwargs = gerberopt.std_func.kwargs
    std_vec = vec(std_func(x, std_args...; std_kwargs...))

    mtx = if normalise
        _sb0_norm(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _sb0(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(mtx, gerberopt.posdef; msg = "SB0 Correlation ", cov_flag = false)

    return mtx, Symmetric(mtx .* (std_vec * transpose(std_vec)), :U)
end

function _sb1_norm(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)

    @inbounds for j ∈ 1:N
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(x))
            pos = zero(eltype(x))
            nn = zero(eltype(x))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (x[k, i] - mui) / sigmai
                xj = (x[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, zero(eltype(x)), zero(eltype(x)),
                                     one(eltype(x)), one(eltype(x)), c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, zero(eltype(x)), zero(eltype(x)),
                                     one(eltype(x)), one(eltype(x)), c1, c2, c3, n)
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += _sb_delta(xi, xj, zero(eltype(x)), zero(eltype(x)),
                                    one(eltype(x)), one(eltype(x)), c1, c2, c3, n)
                end
            end
            den = (pos + neg + nn)
            mtx[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

function _sb1(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)
    @inbounds for j ∈ 1:N
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(x))
            pos = zero(eltype(x))
            nn = zero(eltype(x))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = x[k, i]
                xj = x[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                end
            end
            den = (pos + neg + nn)
            mtx[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

"""
```
sb1(x::AbstractMatrix, gerberopt::GerberOpt = GerberOpt(;), sbopt::SBOpt = SBOpt(;))
```
"""
function sb1(x::AbstractMatrix, gerberopt::GerberOpt = GerberOpt(;),
             sbopt::SBOpt = SBOpt(;))
    c1 = sbopt.c1
    c2 = sbopt.c2
    c3 = sbopt.c3
    n = sbopt.n

    threshold = gerberopt.threshold
    normalise = gerberopt.normalise

    mean_func = gerberopt.mean_func.func
    mean_args = gerberopt.mean_func.args
    mean_kwargs = gerberopt.mean_func.kwargs
    mean_vec = vec(mean_func(x, mean_args...; mean_kwargs...))

    std_func = gerberopt.std_func.func
    std_args = gerberopt.std_func.args
    std_kwargs = gerberopt.std_func.kwargs
    std_vec = vec(std_func(x, std_args...; std_kwargs...))

    mtx = if normalise
        _sb1_norm(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _sb1(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(mtx, gerberopt.posdef; msg = "SB1 Correlation ", cov_flag = false)

    return mtx, Symmetric(mtx .* (std_vec * transpose(std_vec)), :U)
end

function _gerbersb0_norm(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)

    @inbounds for j ∈ 1:N
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(x))
            pos = zero(eltype(x))
            cneg = 0
            cpos = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (x[k, i] - mui) / sigmai
                xj = (x[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, zero(eltype(x)), zero(eltype(x)),
                                     one(eltype(x)), one(eltype(x)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, zero(eltype(x)), zero(eltype(x)),
                                     one(eltype(x)), one(eltype(x)), c1, c2, c3, n)
                    cneg += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            den = (tpos + tneg)
            mtx[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

function _gerbersb0(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)

    @inbounds for j ∈ 1:N
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(x))
            pos = zero(eltype(x))
            cneg = 0
            cpos = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = x[k, i]
                xj = x[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            den = (tpos + tneg)
            mtx[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

"""
```
gerbersb0(x::AbstractMatrix, gerberopt::GerberOpt = GerberOpt(;), sbopt::SBOpt = SBOpt(;))
```
"""
function gerbersb0(x::AbstractMatrix, gerberopt::GerberOpt = GerberOpt(;),
                   sbopt::SBOpt = SBOpt(;))
    c1 = sbopt.c1
    c2 = sbopt.c2
    c3 = sbopt.c3
    n = sbopt.n

    threshold = gerberopt.threshold
    normalise = gerberopt.normalise

    mean_func = gerberopt.mean_func.func
    mean_args = gerberopt.mean_func.args
    mean_kwargs = gerberopt.mean_func.kwargs
    mean_vec = vec(mean_func(x, mean_args...; mean_kwargs...))

    std_func = gerberopt.std_func.func
    std_args = gerberopt.std_func.args
    std_kwargs = gerberopt.std_func.kwargs
    std_vec = vec(std_func(x, std_args...; std_kwargs...))

    mtx = if normalise
        _gerbersb0_norm(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _gerbersb0(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(mtx, gerberopt.posdef; msg = "Gerber_SB0 Correlation ", cov_flag = false)

    return mtx, Symmetric(mtx .* (std_vec * transpose(std_vec)), :U)
end

function _gerbersb1_norm(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)
    @inbounds for j ∈ 1:N
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(x))
            pos = zero(eltype(x))
            nn = zero(eltype(x))
            cneg = 0
            cpos = 0
            cnn = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (x[k, i] - mui) / sigmai
                xj = (x[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, zero(eltype(x)), zero(eltype(x)),
                                     one(eltype(x)), one(eltype(x)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, zero(eltype(x)), zero(eltype(x)),
                                     one(eltype(x)), one(eltype(x)), c1, c2, c3, n)
                    cneg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += _sb_delta(xi, xj, zero(eltype(x)), zero(eltype(x)),
                                    one(eltype(x)), one(eltype(x)), c1, c2, c3, n)
                    cnn += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            tnn = nn * cnn
            den = (tpos + tneg + tnn)
            mtx[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

function _gerbersb1(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    T, N = size(x)
    mtx = Matrix{eltype(x)}(undef, N, N)
    @inbounds for j ∈ 1:N
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(x))
            pos = zero(eltype(x))
            nn = zero(eltype(x))
            cneg = 0
            cpos = 0
            cnn = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = x[k, i]
                xj = x[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cnn += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            tnn = nn * cnn
            den = (tpos + tneg + tnn)
            mtx[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(x))
            end
        end
    end

    mtx .= Symmetric(mtx, :U)

    return mtx
end

"""
```
gerbersb1(x::AbstractMatrix, gerberopt::GerberOpt = GerberOpt(;), sbopt::SBOpt = SBOpt
```
"""
function gerbersb1(x::AbstractMatrix, gerberopt::GerberOpt = GerberOpt(;),
                   sbopt::SBOpt = SBOpt(;))
    c1 = sbopt.c1
    c2 = sbopt.c2
    c3 = sbopt.c3
    n = sbopt.n

    threshold = gerberopt.threshold
    normalise = gerberopt.normalise

    mean_func = gerberopt.mean_func.func
    mean_args = gerberopt.mean_func.args
    mean_kwargs = gerberopt.mean_func.kwargs
    mean_vec = vec(mean_func(x, mean_args...; mean_kwargs...))

    std_func = gerberopt.std_func.func
    std_args = gerberopt.std_func.args
    std_kwargs = gerberopt.std_func.kwargs
    std_vec = vec(std_func(x, std_args...; std_kwargs...))

    mtx = if normalise
        _gerbersb1_norm(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    else
        _gerbersb1(x, mean_vec, std_vec, threshold, c1, c2, c3, n)
    end

    posdef_fix!(mtx, gerberopt.posdef; msg = "Gerber_SB1 Correlation ", cov_flag = false)

    return mtx, Symmetric(mtx .* (std_vec * transpose(std_vec)), :U)
end

function cov_returns(x::AbstractMatrix; iters::Integer = 5, len::Integer = 10,
                     rng = Random.default_rng(), seed::Union{Nothing, <:Integer} = nothing)
    Random.seed!(rng, seed)

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
                 target_ret::Union{Real, <:AbstractVector{<:Real}} = 0.0)
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
                 target_ret::Union{Real, <:AbstractVector{<:Real}} = 0.0)
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

function posdef_nearest(mtx::AbstractMatrix, method = NCM.Newton; cov_flag = true)
    _mtx = if cov_flag
        s = sqrt.(diag(mtx))
        corr = cov2cor(mtx)
        NCM.nearest_cor!(corr, method)
        cor2cov(corr, s)
    else
        NCM.nearest_cor(mtx, method)
    end

    return any(.!isfinite.(_mtx)) ? mtx : _mtx
end

function _optimize_posdef_psd(model, solvers::AbstractDict)
    term_status = termination_status(model)
    solvers_tried = Dict()

    for (key, val) ∈ solvers
        if haskey(val, :solver)
            set_optimizer(model, val[:solver])
        end

        if haskey(val, :params)
            for (attribute, value) ∈ val[:params]
                set_attribute(model, attribute, value)
            end
        end
        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, key => Dict(:jump_error => jump_error))
            continue
        end

        term_status = termination_status(model)

        if term_status ∈ ValidTermination
            break
        end

        push!(solvers_tried,
              key => Dict(:objective_val => objective_value(model),
                          :term_status => term_status,
                          :params => haskey(val, :params) ? val[:params] : missing))
    end

    return solvers_tried
end

function posdef_psd(mtx::AbstractMatrix, solvers::AbstractDict; cov_flag = true)
    _mtx = if cov_flag
        s = sqrt.(diag(mtx))
        cov2cor(mtx)
    else
        copy(mtx)
    end

    n = size(_mtx, 1)
    v = vec(_mtx)
    q = -v
    r = 0.5 * dot(v, v)

    model = JuMP.Model()
    set_string_names_on_creation(model, false)

    @variable(model, X[1:n, 1:n] ∈ PSDCone())
    x = vec(X)
    @objective(model, Min, 0.5 * dot(x, x) + dot(q, x) + r)
    for i ∈ 1:n
        @constraint(model, X[i, i] == 1.0)
    end

    solvers_tried = _optimize_posdef_psd(model, solvers)
    term_status = termination_status(model)

    if term_status ∉ ValidTermination
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.posdef_psd))"
        @warn("$funcname: model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        return mtx
    end

    _mtx .= if cov_flag
        cor2cov(value.(X), s)
    else
        value.(X)
    end

    return any(.!isfinite.(_mtx)) ? mtx : _mtx
end

"""
```
posdef_fix!(mtx::AbstractMatrix, opt::PosdefFixOpt = PosdefFixOpt(;), msg::String = "")
```
"""
function posdef_fix!(mtx::AbstractMatrix, opt::PosdefFixOpt = PosdefFixOpt(;);
                     msg::String = "", cov_flag = true)
    method = opt.method

    if method == :None || isposdef(mtx)
        return nothing
    end

    @smart_assert(method ∈ PosdefFixMethods)

    func = opt.genfunc.func
    args = opt.genfunc.args
    kwargs = opt.genfunc.kwargs
    solvers = opt.solvers

    _mtx = if method == :Nearest
        posdef_nearest(mtx, args...; cov_flag = cov_flag, kwargs...)
    elseif method == :SDP
        posdef_psd(mtx, solvers; cov_flag = cov_flag)
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
```
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

"""
```
_denoise_logo_mtx(T::Integer, N::Integer, mtx::AbstractMatrix,
                  opt::Union{CovOpt, CorOpt, KurtOpt, BLOpt}, mtx_name::Symbol = :cov)
```
"""
function _denoise_logo_mtx(T::Integer, N::Integer, mtx::AbstractMatrix,
                           opt::Union{CovOpt, CorOpt, KurtOpt, SkewOpt, BLOpt},
                           mtx_name::Symbol = :cov, cov_flag = true)
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
    elseif mtx_name == :skew
        msg = "Skewness "
        msg2 = ""
    elseif mtx_name == :sskew
        msg = "Semi Skewness "
        msg2 = ""
    elseif mtx_name == :bl_cov
        msg = "Black-Litterman Covariance "
        msg2 = "\n\t Try some different asset views."
    elseif mtx_name == :a_bl_cov
        msg = "Augmented Black-Litterman Covariance "
        msg2 = "\n\t Try a different combination of asset views, factor views, and/or loadings matrix."
    elseif mtx_name == :af_bl_cov
        msg = "Augmented Black-Litterman Covariance with no asset views "
        msg2 = "\n\t Try a different combination of factor views, and/or loadings matrix."
    end

    mtx = denoise_cov(mtx, T / N, opt.denoise)

    posdef_fix!(mtx, opt.posdef; msg = msg, cov_flag = cov_flag)

    jlogo = opt.jlogo
    if jlogo.flag
        try
            corr = cov_flag ? cov2cor(mtx) : mtx
            distance = jlogo.dist
            dist_method = distance.method
            dargs = distance.args
            dkwargs = distance.kwargs
            dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
            if size(dist, 1) != size(dist, 2)
                dist = reshape(dist, size(corr))
            end

            genfunc = jlogo.genfunc
            func = genfunc.func
            args = genfunc.args
            kwargs = genfunc.kwargs
            corr = func(corr, dist, args...; kwargs...)
            separators, cliques = PMFG_T2s(corr, 4)[3:4]

            mtx .= J_LoGo(mtx, separators, cliques) \ I
        catch SingularException
            throw(ErrorException("$msg matrix is singular = $(SingularException). Please try one or a combination of the following:\n\t* Set opt.posdef.method = $(opt.posdef.method), to a different method from $PosdefFixMethods.\n\t* Set denoise = true.\n\t* Try both approaches at the same time.$(msg2)"))
        end
        posdef_fix!(mtx, opt.posdef; msg = "J-LoGo $msg ", cov_flag = cov_flag)
    end

    return mtx
end

"""
```
covar_mtx(returns::AbstractMatrix, opt::CovOpt = CovOpt(;))
```

Compute the covariance matrix. See [`gerber0`](@ref), [`gerber1`](@ref), [`gerber2`](@ref), [`sb0`](@ref), [`sb1`](@ref), [`gerbersb0`](@ref), [`gerbersb1`](@ref), [`_denoise_logo_mtx`](@ref), [`posdef_fix!`](@ref).

# Inputs

  - `returns`: `T×N` matrix of returns, where `T` is the number of returns observations, and `N` is the number of assets or factors.
  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the covariance matrix.

# Outputs

  - `sigma`: `N×N` covariance matrix, where `N` is the number of assets or factors.
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
        gerber0(returns, opt.gerber)[2]
    elseif method == :Gerber1
        gerber1(returns, opt.gerber)[2]
    elseif method == :Gerber2
        gerber2(returns, opt.gerber)[2]
    elseif method == :SB0
        sb0(returns, opt.gerber, opt.sb)[2]
    elseif method == :SB1
        sb1(returns, opt.gerber, opt.sb)[2]
    elseif method == :Gerber_SB0
        gerbersb0(returns, opt.gerber, opt.sb)[2]
    elseif method == :Gerber_SB1
        gerbersb1(returns, opt.gerber, opt.sb)[2]
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

    mtx = issymmetric(mtx) ? mtx : Symmetric(mtx, opt.uplo)

    return mtx
end

"""
```
mean_vec(returns::AbstractMatrix, opt::MuOpt = MuOpt(;))
```

Compute the expected returns vector for a returns series.

# Inputs

  - `returns`: `T×N` matrix of returns, where `T` is the number of returns observations, and `N` is the number of assets or factors.
  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the expected returns vector.

# Outputs

  - `mu`: `N×1` vector of expected returns, where `N` is the number of assets or factors.
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
```
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

function coskew_mtx(returns::AbstractMatrix, mu::AbstractVector, opt::SkewOpt = SkewOpt(;))
    custom_skew = opt.estimation.custom_skew
    T, N = size(returns)
    if isnothing(custom_skew)
        skew = coskew(returns, transpose(mu))
    else
        skew = custom_skew
    end
    V = zeros(eltype(skew), N, N)
    for i ∈ 1:N
        j = (i - 1) * N + 1
        k = i * N
        vals, vecs = eigen(skew[:, j:k])
        vals = clamp.(real.(vals), -Inf, 0) .+ clamp.(imag.(vals), -Inf, 0)im
        V .-= real(vecs * Diagonal(vals) * transpose(vecs))
    end

    target_ret = opt.estimation.target_ret
    custom_sskew = opt.estimation.custom_sskew
    if isnothing(custom_sskew)
        sskew = scoskew(returns, transpose(mu), target_ret)
    else
        sskew = custom_sskew
    end
    SV = zeros(eltype(sskew), N, N)
    for i ∈ 1:N
        j = (i - 1) * N + 1
        k = i * N
        vals, vecs = eigen(sskew[:, j:k])
        vals = clamp.(real.(vals), -Inf, 0) .+ clamp.(imag.(vals), -Inf, 0)im
        SV .-= real(vecs * Diagonal(vals) * transpose(vecs))
    end

    return skew, V, sskew, SV
end

"""
```
cor_dist_mtx(returns::AbstractMatrix, opt::CorOpt = CorOpt(;))
```
"""
function cor_dist_mtx(returns::AbstractMatrix, opt::CorOpt = CorOpt(;))
    method = opt.method
    T, N = size(returns)

    if method ∈ (:Abs_Pearson, :Abs_Semi_Pearson, :Abs_Spearman, :Abs_Kendall)
        if hasproperty(opt.dist.method, :absolute)
            opt.dist.method.absolute = true
        end
        if hasproperty(opt.jlogo.dist.method, :absolute)
            opt.jlogo.dist.method.absolute = true
        end
    else
        if hasproperty(opt.dist.method, :absolute)
            opt.dist.method.absolute = false
        end
        if hasproperty(opt.jlogo.dist.method, :absolute)
            opt.jlogo.dist.method.absolute = false
        end
    end

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
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Spearman
        corr = corspearman(returns)
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Kendall
        corr = corkendall(returns)
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
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
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        # dist = sqrt.(clamp!(1 .- corr, 0, 1))
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Abs_Spearman
        corr = abs.(corspearman(returns))
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        # dist = sqrt.(clamp!(1 .- corr, 0, 1))
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Abs_Kendall
        corr = abs.(corkendall(returns))
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        # dist = sqrt.(clamp!(1 .- corr, 0, 1))
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Gerber0
        corr = gerber0(returns, opt.gerber)[1]
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Gerber1
        corr = gerber1(returns, opt.gerber)[1]
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Gerber2
        corr = gerber2(returns, opt.gerber)[1]
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :SB0
        corr = sb0(returns, opt.gerber, opt.sb)[1]
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :SB1
        corr = sb1(returns, opt.gerber, opt.sb)[1]
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Gerber_SB0
        corr = gerbersb0(returns, opt.gerber, opt.sb)[1]
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Gerber_SB1
        corr = gerbersb1(returns, opt.gerber, opt.sb)[1]
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Distance
        corr = Matrix(cordistance(returns, opt.distcor))
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
    elseif method == :Mutual_Info
        corr, dist = mut_var_info_mtx(returns, opt.estimation.bins_info)
    elseif method == :Tail
        corr = Matrix(ltdi_mtx(returns, opt.estimation.alpha))
        corr = _denoise_logo_mtx(T, N, corr, opt, :cor, true)
        dist = -log.(corr)
    elseif method == :Cov_to_Cor
        estimation = opt.estimation
        sigma = estimation.sigma
        corr = cov2cor(sigma)
        # dist_func = estimation.dist_genfunc.func
        # dist_args = estimation.dist_genfunc.args
        # dist_kwargs = estimation.dist_genfunc.kwargs
        # dist = dist_func(corr, dist_args...; dist_kwargs...)
        dist_method = opt.dist.method
        dargs = opt.dist.args
        dkwargs = opt.dist.kwargs
        dist = pairwise(dist_method, corr, one(eltype(corr)), dargs...; dkwargs...)
        if size(dist, 1) != size(dist, 2)
            dist = reshape(dist, size(corr))
        end
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

"""
```
covar_mtx_mean_vec(returns::AbstractMatrix, cov_opt::CovOpt = CovOpt(;),
                   mu_opt::MuOpt = MuOpt(;))
```

Compute the expected returns vector and covariance matrix. See [`covar_mtx`](@ref) and [`mean_vec`](@ref).

# Inputs

  - `returns`: `T×N` matrix of returns, where `T` is the number of returns observations, and `N` is the number of assets or factors.
  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the covariance matrix.
  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the expected returns vector.

# Outputs

  - `sigma`: `N×N` covariance matrix, where `N` is the number of assets or factors.
  - `mu`: `N×1` vector of expected returns, where `N` is the number of assets or factors.
"""
function covar_mtx_mean_vec(returns::AbstractMatrix; cov_opt::CovOpt = CovOpt(;),
                            mu_opt::MuOpt = MuOpt(;))
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
```
asset_statistics!(portfolio::AbstractPortfolio; calc_cov::Bool = true, calc_mu::Bool = true,
                  calc_kurt::Bool = true, calc_skew::Bool=true, calc_cor::Bool = true,
                  cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;),
                  kurt_opt::KurtOpt = KurtOpt(;), cor_opt::CorOpt = CorOpt(;))
```

Compute the asset statistics for a given `portfolio` in-place. See [`covar_mtx`](@ref), [`mean_vec`](@ref), [`cokurt_mtx`](@ref), [`cor_dist_mtx`](@ref).

Depending on conditions, modifies:

  - `portfolio.mu`
  - `portfolio.cov`
  - `portfolio.kurt`
  - `portfolio.skurt`
  - `portfolio.cor`
  - `portfolio.dist`
  - `mu_opt.method`

# Inputs

  - `portfolio`: instance of [`Portfolio`](@ref) or [`HCPortfolio`](@ref).

## Flags

  - `calc_cov`:

      + `true`: compute and set the covariance matrix.

  - `calc_mu`:

      + `true`: compute and set the expected returns vector.

          * `mu_opt.method ∈ (:JS, :BS, :BOP, :CAPM)`: require the covariance matrix, so it will be computed and `mu_opt.sigma` will be set to the covariance matrix. `portfolio.cov` will only be modified when `calc_cov == true`.
  - `calc_kurt`:

      + `true`: compute and set the cokurtosis and semi cokurtosis matrices.
  - `calc_skew`:

      + `true`: compute and set the coskewness and semi coskewness matrices.

      + `calc_cor`:

          * `isa(portfolio, HCPortfolio)`:

              - `true`: compute and set the correlation and distance matrices.

## Options

  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the covariance matrix.
  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the expected returns vector.
  - `kurt_opt`: instance of [`KurtOpt`](@ref), defines how the cokurtosis and semi cokurtoes are computed.
  - `cor_opt`: instance of [`CorOpt`](@ref), defines how the correlation and distance matrices are computed.
"""
function asset_statistics!(portfolio::AbstractPortfolio; calc_cov::Bool = true,
                           calc_mu::Bool = true, calc_kurt::Bool = true, calc_skew = true,
                           calc_cor::Bool = true, cov_opt::CovOpt = CovOpt(;),
                           mu_opt::MuOpt = MuOpt(;), kurt_opt::KurtOpt = KurtOpt(;),
                           skew_opt::SkewOpt = SkewOpt(;), cor_opt::CorOpt = CorOpt(;))
    returns = portfolio.returns

    sigma = nothing
    if calc_mu
        mu_method = mu_opt.method
        if mu_method == :CAPM
            mkt_ret = mu_opt.mkt_ret
            returns = if isnothing(mkt_ret)
                hcat(returns, mean(returns; dims = 2))
            else
                hcat(returns, mkt_ret)
            end
        end

        if mu_opt.method ∈ (:JS, :BS, :BOP, :CAPM)
            sigma = covar_mtx(returns, cov_opt)
            mu_opt.sigma = sigma
        end

        mu = mean_vec(returns, mu_opt)

        if mu_method == :CAPM
            sigma = sigma[1:(end - 1), 1:(end - 1)]
            returns = returns[:, 1:(end - 1)]
        end

        portfolio.mu = mu
    end

    if calc_cov
        if isnothing(sigma)
            sigma = covar_mtx(returns, cov_opt)
        end
        portfolio.cov = sigma
    end

    if calc_kurt
        portfolio.kurt, portfolio.skurt, portfolio.L_2, portfolio.S_2 = cokurt_mtx(returns,
                                                                                   portfolio.mu,
                                                                                   kurt_opt)
    end

    if calc_skew
        portfolio.skew, portfolio.V, portfolio.sskew, portfolio.SV = coskew_mtx(returns, mu,
                                                                                skew_opt)
    end

    # Type specific
    if isa(portfolio, HCPortfolio) && calc_cor
        cor_opt.estimation.sigma = portfolio.cov
        portfolio.cor, portfolio.dist = cor_dist_mtx(returns, cor_opt)
        portfolio.cor_method = cor_opt.method
    end

    return nothing
end

"""
```
commutation_matrix(x::AbstractMatrix)
```

Generates the [commutation matrix](https://en.wikipedia.org/wiki/Commutation_matrix) for `x`.

# Inputs

  - `x`: matrix.

# Outputs

  - `y`: commutation matrix.
"""
function commutation_matrix(x::AbstractMatrix)
    m, n = size(x)
    mn = m * n
    row = 1:mn
    col = vec(transpose(reshape(row, m, n)))
    data = range(; start = 1, stop = 1, length = mn)
    com = sparse(row, col, data, mn, mn)
    return com
end

"""
```
gen_bootstrap(returns::AbstractMatrix, cov_opt::CovOpt, mu_opt::MuOpt,
              method::Symbol = :Stationary, n_sim::Integer = 3_000, block_size::Integer = 3,
              seed::Union{<:Integer, Nothing} = nothing)
```

Simulate returns series using bootstrapping with [arch](https://github.com/bashtage/arch/?tab=readme-ov-file).

# Inputs

  - `returns`: `T×N` matrix of returns, where `T` is the number of returns observations, and `N` the number of assets.
  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the covariance matrices of the bootstrapping simulations, as well as fixing non-positive definite matrices.
  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the expected returns vectors of the bootstrapping simulations.
  - `kind`: bootstrapping method from [`BootstrapMethods`](@ref).
  - `n_sim`: number of simulations.
  - `block_size`: average block size to use.
  - `seed`: random number generator seed for bootstrapping.
"""
function gen_bootstrap(returns::AbstractMatrix, cov_opt::CovOpt, mu_opt::MuOpt,
                       method::Symbol = :Stationary, n_sim::Integer = 3_000,
                       block_size::Integer = 3, seed::Union{<:Integer, Nothing} = nothing)
    @smart_assert(method ∈ BootstrapMethods)

    mus = Vector{Vector{eltype(returns)}}(undef, 0)
    sizehint!(mus, n_sim)
    covs = Vector{Matrix{eltype(returns)}}(undef, 0)
    sizehint!(covs, n_sim)

    bootstrap_func = if method == :Stationary
        pyimport("arch.bootstrap").StationaryBootstrap
    elseif method == :Circular
        pyimport("arch.bootstrap").CircularBlockBootstrap
    elseif method == :Moving
        pyimport("arch.bootstrap").MovingBlockBootstrap
    end

    gen = bootstrap_func(block_size, returns; seed = seed)
    for data ∈ gen.bootstrap(n_sim)
        A = data[1][1]
        sigma, mu = covar_mtx_mean_vec(A; cov_opt = cov_opt, mu_opt = mu_opt)
        push!(mus, mu)
        push!(covs, sigma)
    end

    return mus, covs
end

"""
```
vec_of_vecs_to_mtx(x::AbstractVector{<:AbstractArray})
```

Turns a vector of arrays into a matrix.

# Inputs

  - `x`: vector of arrays.

# Outputs

  - `y`: matrix.
"""
function vec_of_vecs_to_mtx(x::AbstractVector{<:AbstractArray})
    return vcat(transpose.(x)...)
end

"""
```
wc_statistics!(portfolio::Portfolio, opt::WCOpt = WCOpt(;))
```

Compute worst case statistics for a given `portfolio` in-place. See [`gen_bootstrap`](@ref), [`vec_of_vecs_to_mtx`](@ref), [`posdef_fix!`](@ref), and [`commutation_matrix`](@ref).

Depending on conditions, modifies:

  - `portfolio.cov_l`
  - `portfolio.cov_u`
  - `portfolio.d_mu`
  - `portfolio.cov_mu`
  - `portfolio.cov_sigma`
  - `portfolio.k_mu`
  - `portfolio.k_sigma`

# Inputs

  - `portfolio`: instance of [`Portfolio`](@ref) or [`HCPortfolio`](@ref).

## Options

  - `opt`: instance of [`WCOpt`](@ref), defines the parameters for computing the worst case statistics.
"""
function wc_statistics!(portfolio::Portfolio, opt::WCOpt = WCOpt(;))
    calc_box = opt.calc_box
    calc_ellipse = opt.calc_ellipse

    if !(calc_box || calc_ellipse)
        return nothing
    end

    diagonal = opt.diagonal
    box = opt.box
    ellipse = opt.ellipse
    k_mu_method = opt.k_mu_method
    k_sigma_method = opt.k_sigma_method
    dcov = opt.dcov
    dmu = opt.dmu
    q = opt.q
    rng = opt.rng
    seed = opt.seed
    n_sim = opt.n_sim
    block_size = opt.block_size
    mu_opt = opt.mu_opt
    cov_opt = opt.cov_opt
    posdef = cov_opt.posdef

    returns = portfolio.returns
    T, N = size(returns)

    sigma, mu = covar_mtx_mean_vec(returns; cov_opt = cov_opt, mu_opt = mu_opt)

    if calc_box
        if box == :Stationary || box == :Circular || box == :Moving
            mus, covs = gen_bootstrap(returns, cov_opt, mu_opt, box, n_sim, block_size,
                                      seed)

            mu_s = vec_of_vecs_to_mtx(mus)
            mu_l = [quantile(mu_s[:, i], q / 2) for i ∈ 1:N]
            mu_u = [quantile(mu_s[:, i], 1 - q / 2) for i ∈ 1:N]

            cov_s = vec_of_vecs_to_mtx(vec.(covs))
            cov_l = reshape([quantile(cov_s[:, i], q / 2) for i ∈ 1:(N * N)], N, N)
            cov_u = reshape([quantile(cov_s[:, i], 1 - q / 2) for i ∈ 1:(N * N)], N, N)

            d_mu = (mu_u - mu_l) / 2
        elseif box == :Normal
            Random.seed!(rng, seed)

            cov_mu = sigma / T
            d_mu = cquantile(Normal(), q / 2) * sqrt.(diag(cov_mu))

            covs = vec_of_vecs_to_mtx(vec.(rand(Wishart(T, cov_mu), n_sim)))
            cov_l = reshape([quantile(covs[:, i], q / 2) for i ∈ 1:(N * N)], N, N)
            cov_u = reshape([quantile(covs[:, i], 1 - q / 2) for i ∈ 1:(N * N)], N, N)
        elseif box == :Delta
            d_mu = dmu * abs.(mu)
            cov_l = sigma - dcov * abs.(sigma)
            cov_u = sigma + dcov * abs.(sigma)
        end

        posdef_fix!(cov_l, posdef; msg = "WC cov_l ")
        posdef_fix!(cov_u, posdef; msg = "WC cov_u ")

        portfolio.cov_l = cov_l
        portfolio.cov_u = cov_u
        portfolio.d_mu = d_mu
    end

    if calc_ellipse
        if ellipse == :Stationary || ellipse == :Circular || ellipse == :Moving
            mus, covs = gen_bootstrap(returns, cov_opt, mu_opt, ellipse, n_sim, block_size,
                                      seed)

            A_mu = vec_of_vecs_to_mtx([mu_s .- mu for mu_s ∈ mus])
            cov_mu = covar_mtx(A_mu, cov_opt)

            A_sigma = vec_of_vecs_to_mtx([vec(cov_s) .- vec(sigma) for cov_s ∈ covs])
            cov_sigma = covar_mtx(A_sigma, cov_opt)
        elseif ellipse == :Normal
            Random.seed!(rng, seed)

            A_mu = transpose(rand(MvNormal(mu, sigma), n_sim))
            if !calc_box || calc_box && box != :Normal
                cov_mu = sigma / T
                covs = vec_of_vecs_to_mtx(vec.(rand(Wishart(T, cov_mu), n_sim)))
            end
            A_sigma = covs .- transpose(vec(sigma))

            K = commutation_matrix(sigma)
            cov_sigma = T * (I + K) * kron(cov_mu, cov_mu)
        end

        posdef_fix!(cov_mu, posdef; msg = "WC cov_mu ")
        posdef_fix!(cov_sigma, posdef; msg = "WC cov_sigma ")

        if diagonal
            cov_mu = Diagonal(cov_mu)
            cov_sigma = Diagonal(cov_sigma)
        end

        k_mu = if k_mu_method == :Normal
            k_mus = diag(A_mu * (cov_mu \ I) * transpose(A_mu))
            sqrt(quantile(k_mus, 1 - q))
        elseif k_mu_method == :General
            sqrt((1 - q) / q)
        else
            k_mu_method
        end

        k_sigma = if k_sigma_method == :Normal
            k_sigmas = diag(A_sigma * (cov_sigma \ I) * transpose(A_sigma))
            sqrt(quantile(k_sigmas, 1 - q))
        elseif k_sigma_method == :General
            sqrt((1 - q) / q)
        else
            k_sigma_method
        end

        portfolio.cov_mu = cov_mu
        portfolio.cov_sigma = cov_sigma
        portfolio.k_mu = k_mu
        portfolio.k_sigma = k_sigma
    end

    return nothing
end

"""
```
forward_regression(x::DataFrame, y::AbstractVector, criterion::Symbol = :pval,
                   threshold::Real = 0.05)
```

Select the factors that best estimate the model based on forward regression.

# Inputs

  - `x`: is the `T×N` Dataframe of factor returns, where the column names are the factor names, `T` is the number of returns observations, and `N` the number of factors.

  - `y`: is the `T×1` vector of returns for an asset.
  - `criterion`: one of [`RegCriteria`](@ref) that decides what criterion to use when selecting the most significant factors.
  - `threshold`:

      + `criterion == :pval`: a factor is considered significant if its p-value is lower than `threshold`. If no factor has a p-value lower than `threshold`, selects the factor with the lowest p-value.

# Outputs

  - `features`: is the `C×1` vector of significant factors, where `C` is the number of significant factors.
"""
function forward_regression(x::DataFrame, y::AbstractVector, criterion::Symbol = :pval,
                            threshold::Real = 0.05)
    @smart_assert(criterion ∈ RegCriteria)

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

"""
```
backward_regression(x::DataFrame, y::AbstractVector, criterion::Symbol = :pval,
                    threshold::Real = 0.05)
```

Select the factors that best estimate the model based on backward regression. This tends to be more robust than [`forward_regression`](@ref).

# Inputs

  - `x`: is the `T×N` Dataframe of factor returns, where the column names are the factor names, `T` is the number of returns observations, and `N` the number of factors.

  - `y`: is the `T×1` vector of returns for an asset.
  - `criterion`: one of [`RegCriteria`](@ref) that decides what criterion to use when selecting the most significant factors.
  - `threshold`:

      + `criterion == :pval`: a factor is considered significant if its p-value is lower than `threshold`. If no factor has a p-value lower than `threshold`, selects the factor with the lowest p-value.

# Outputs

  - `features`: is the `C×1` vector of significant factors, where `C` is the number of significant factors.
"""
function backward_regression(x::DataFrame, y::AbstractVector, criterion::Symbol = :pval,
                             threshold::Real = 0.05)
    @smart_assert(criterion ∈ RegCriteria)

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

function _prep_mvr(x::DataFrame, opt::MVROpt = MVROpt(;))
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

    return X, x1, Vp
end

function _mvr(X::AbstractMatrix, x1::AbstractMatrix, Vp::AbstractMatrix, y::AbstractVector,
              opt::MVROpt = MVROpt(;))
    mean_genfunc = opt.mean_genfunc
    std_genfunc = opt.std_genfunc

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

"""
```
stepwise_regression(x::DataFrame, y::DataFrame, opt::LoadingsOpt = LoadingsOpt(;))
```

Perform forward or backward stepwise regression. See [`forward_regression`](@ref), and [`backward_regression`](@ref).

# Inputs

  - `x`: is the `T×Nf` Dataframe of factor returns, where the column names are the factor names, `T` is the number of returns observations, and `Nf` the number of factors.
  - `y`: is the `T×Na` Dataframe of asset returns, where the column names are the factor names, `T` is the number of returns observations, and `Na` the number of assets.

## Options

  - `loadings_opt`: instance of [`LoadingsOpt`](@ref), defines the parameters for computing the loadings matrix.

# Outputs

  - `B`: is the `(Na+2)×Nf` loadings matrix as a Dataframe, where `Na` is the number of assets, `Nf` the number of factors. The two extra columns are the optional columns for `B` as described in [`FactorOpt`](@ref).
"""
function stepwise_regression(x::DataFrame, y::DataFrame, opt::LoadingsOpt = LoadingsOpt(;))
    features = names(x)
    rows = ncol(y)
    cols = ncol(x) + 1

    N = nrow(y)
    ovec = ones(N)

    loadings = zeros(rows, cols)

    criterion = opt.criterion
    threshold = opt.threshold
    method = opt.method

    for i ∈ 1:rows
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
    end

    return hcat(DataFrame(; tickers = names(y)), DataFrame(loadings, ["const"; features]))
end

"""
```
mv_regression(x::DataFrame, y::DataFrame, opt::MVROpt = MVROpt(;))
```

Select the factors that best estimate the model based on multivariate regression. This tends to be more robust than [`backward_regression`](@ref).

# Inputs

  - `x`: is the `T×N` Dataframe of factor returns, where the column names are the factor names, `T` is the number of returns observations, and `N` the number of factors.
  - `y`: is the `T×N` Dataframe of asset returns.
  - `opt`: instance of [`MVROpt`](@ref) for defining the regression and its parameters.

# Outputs

  - `B`: is the `(Na+2)×Nf` loadings matrix as a Dataframe, where `Na` is the number of assets, `Nf` the number of factors. The two extra columns are the optional columns for `B` as described in [`FactorOpt`](@ref).
"""
function mv_regression(x::DataFrame, y::DataFrame, opt::MVROpt = MVROpt(;))
    features = names(x)
    rows = ncol(y)
    cols = ncol(x) + 1

    loadings = zeros(rows, cols)

    X, x1, Vp = _prep_mvr(x, opt)
    for i ∈ 1:rows
        beta = _mvr(X, x1, Vp, y[!, i], opt)
        loadings[i, :] .= beta
    end

    return hcat(DataFrame(; tickers = names(y)), DataFrame(loadings, ["const"; features]))
end

"""
```
loadings_matrix(x::DataFrame, y::DataFrame, opt::LoadingsOpt = LoadingsOpt(;))
```

Estimate the loadings matrix using regression. See [`stepwise_regression`](@ref), and [`mv_regression`](@ref).

# Inputs

  - `x`: is the `T×Nf` Dataframe of factor returns, where the column names are the factor names, `T` is the number of returns observations, and `Nf` the number of factors.
  - `y`: is the `T×Na` Dataframe of asset returns, where the column names are the factor names, `T` is the number of returns observations, and `Na` the number of assets.

## Options

  - `loadings_opt`: instance of [`LoadingsOpt`](@ref), defines the parameters for computing the loadings matrix.

# Outputs

  - `B`: is the `(Na+2)×Nf` loadings matrix as a Dataframe, where `Na` is the number of assets, `Nf` the number of factors. The two extra columns are the optional columns for `B` as described in [`FactorOpt`](@ref).
"""
function loadings_matrix(x::DataFrame, y::DataFrame, opt::LoadingsOpt = LoadingsOpt(;))
    loadings_matrix = if opt.method ∈ (:FReg, :BReg)
        stepwise_regression(x, y, opt)
    else
        mv_regression(x, y, opt.mvr_opt)
    end

    return loadings_matrix
end

"""
```
risk_factors(x::DataFrame, y::DataFrame; factor_opt::FactorOpt = FactorOpt(;),
             cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;))
```

Estimates the returns matrix, expected returns vector, and covariance matrix of the assets based on the risk factor model [FM1, FM2](@cite). See [`loadings_matrix`](@ref), [`covar_mtx_mean_vec`](@ref), and [`posdef_fix!`](@ref).

```math
\\begin{align*}
\\mathbf{X}_{\\mathrm{FM}} &= \\mathbf{F} \\mathbf{B}^{\\intercal}\\\\
\\bm{\\mu}_{\\mathrm{FM}} &= \\mathbf{B} \\bm{\\mu}_{F}\\\\
\\bm{\\Sigma}_{\\mathrm{FM}} &= \\mathbf{B} \\mathbf{\\Sigma}_{F} \\mathbf{B}^{\\intercal} + \\mathbf{\\Sigma}_{\\epsilon}\\\\
\\mathbf{\\Sigma}_{\\epsilon} &= \\begin{cases}\\mathrm{Diagonal}\\left(\\mathrm{var}\\left(\\mathbf{X} - \\mathbf{X}_{\\mathrm{FM}},\\, \\mathrm{dims} = 1\\right)\\right) &\\quad \\mathrm{if~ error = true}\\\\
\\mathbf{0} &\\quad \\mathrm{if~ error = false}
\\end{cases}
\\end{align*}
```

Where:

  - ``\\mathbf{X}_{\\mathrm{FM}}``: is the `T×Na` matrix of factor adjusted asset returns, where `T` is the number of returns observations, and `Na` the number of assets.
  - ``\\mathbf{F}``: is the `T×Nf` matrix of factor returns, where `T` is the number of returns observations, and `Nf` the number of assets.
  - ``\\mathbf{B}``: is the `Na×Nf` loadings matrix, where `Na` is the number of assets, and `Nf` the number of factors.
  - ``\\bm{\\mu}_{\\mathrm{FM}}``: is the `Na×1` estimated expected asset returns vector computed using the factor model, where `Na` is the number of assets.
  - ``\\bm{\\mu}_{F}``: is the `Nf×1` expected factor returns vector, where `Nf` is the number of factors.
  - ``\\mathbf{\\Sigma}_{\\epsilon}``: is an `Na×Na` diagonal matrix constructed from the variances of the errors between the asset and estimated asset returns using the factor model, where `Na` is the number of assets. The variance is taken over all `T` timestamps of `Na` assets.
  - ``\\mathbf{\\Sigma}_{\\mathrm{FM}}``: is the `Na×Na` estimated asset covariance matrix computed using the factor model, where `Na` is the number of assets.

# Inputs

  - `x`: is the `T×Nf` Dataframe of factor returns, where the column names are the factor names, `T` is the number of returns observations, and `Nf` the number of factors.
  - `y`: is the `T×Na` Dataframe of asset returns, where the column names are the factor names, `T` is the number of returns observations, and `Na` the number of assets.

## Options

  - `factor_opt`: instance of [`FactorOpt`](@ref), defines the parameters for computing the factors and loadings matrix.

      + `isnothing(factor_opt.B)`: the loadings matris is computed internally using `factor_opt.loadings_opt`.

  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the factor covariance matrix, as well as fixing non-positive definite matrices.

      + If the loadings matrix contains the `const` column `cov_opt.posdef.method` is set to `:None` before computing the factor covariance matrix because the constant term makes it impossible to turn it into a positive definite matrix. `cov_opt.posdef.method` is then reset to its original value so it can be used to fix the estimated asset covariance matrix if needed.
  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the expected factor returns vector.

# Outputs

  - `mu`: is the `Na×1` estimated expected asset returns vector computed using the factor model, where `Na` is the number of assets.
  - `sigma`: is the `Na×Na` estimated asset covariance matrix computed using the factor model, where `Na` is the number of assets.
  - `returns`: is the `T×Na` matrix of factor adjusted asset returns, where `T` is the number of returns observations, and `Na` the number of assets.
  - `B`: is the `(Na+c)×Nf` loadings matrix as a Dataframe, where `Na` is the number of assets, `Nf` the number of factors, and `c ∈ (0, 1, 2)` represents the two optional columns as described in [`FactorOpt`](@ref).
"""
function risk_factors(x::DataFrame, y::DataFrame; factor_opt::FactorOpt = FactorOpt(;),
                      cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;))
    B = factor_opt.B

    if isnothing(B)
        B = loadings_matrix(x, y, factor_opt.loadings_opt)
    end
    namesB = names(B)
    old_posdef = nothing
    x1 = if "const" ∈ namesB
        old_posdef = cov_opt.posdef.method
        cov_opt.posdef.method = :None
        [ones(nrow(y)) Matrix(x)]
    else
        Matrix(x)
    end
    B_mtx = Matrix(B[!, setdiff(namesB, ("tickers",))])

    f_cov, f_mu = covar_mtx_mean_vec(x1; cov_opt = cov_opt, mu_opt = mu_opt)

    if !isnothing(old_posdef)
        cov_opt.posdef.method = old_posdef
        f_cov2 = f_cov[2:end, 2:end]
        posdef_fix!(f_cov2, cov_opt.posdef; msg = "Factor Covariance ")
        f_cov[2:end, 2:end] .= f_cov2
    end

    returns = x1 * transpose(B_mtx)
    mu = B_mtx * f_mu

    sigma = if factor_opt.error
        var_genfunc = factor_opt.var_genfunc
        var_func = var_genfunc.func
        var_args = var_genfunc.args
        var_kwargs = var_genfunc.kwargs
        e = Matrix(y) - returns
        S_e = diagm(vec(var_func(e, var_args...; var_kwargs...)))
        B_mtx * f_cov * transpose(B_mtx) + S_e
    else
        B_mtx * f_cov * transpose(B_mtx)
    end

    posdef_fix!(sigma, cov_opt.posdef; msg = "Factor Model Covariance ")

    return mu, sigma, returns, B
end

"""
```
_omega(P, tau_sigma)
```
"""
function _omega(P, tau_sigma)
    return Diagonal(P * tau_sigma * transpose(P))
end

"""
```
_Pi(eq, delta, sigma, w, mu, rf)
```
"""
function _Pi(eq, delta, sigma, w, mu, rf)
    return eq ? delta * sigma * w : mu .- rf
end

"""
```
_mu_cov_w(tau, omega, P, Pi, Q, rf, sigma, delta, T, N, opt, cov_type, cov_flag = true)
```

Internal function for computing the Black Litterman statistics as defined in [`black_litterman`](@ref). See [`_denoise_logo_mtx`](@ref).

# Inputs

  - `tau`: variable of the same name in the Black-Litterman model.
  - `omega`: variable of the same name in the Black-Litterman model.
  - `P`: variable of the same name in the Black-Litterman model.
  - `Pi`: variable of the same name in the Black-Litterman model.
  - `Q`: variable of the same name in the Black-Litterman model.
  - `rf`: variable of the same name in the Black-Litterman model.
  - `sigma`: variable of the same name in the Black-Litterman model.
  - `delta`: variable of the same name in the Black-Litterman model.
  - `T`: variable of the same name in the Black-Litterman model.
  - `N`: variable of the same name in the Black-Litterman model.
  - `opt`: any valid instance of `opt` for [`_denoise_logo_mtx`](@ref).
  - `cov_type`: any valid value from [`DenoiseLoGoNames`](@ref).
  - `cov_flag`: whether the matrix is a covariance matrix or not.

# Outputs

  - `mu`: asset expected returns vector obtained via the Black-Litterman model.
  - `cov_mtx`: asset covariance matrix obtained via the Black-Litterman model.
  - `w`: asset weights obtained via the Black-Litterman model.
  - `Pi_`: equilibrium excess returns after being adjusted by the views.
"""
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

"""
```
black_litterman(returns::AbstractMatrix, P::AbstractMatrix, Q::AbstractVector,
                w::AbstractVector; cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;),
                bl_opt::BLOpt = BLOpt(;))
```

Estimates the expected returns vector and covariance matrix based on the Black-Litterman model [BL1, BL2](@cite). See [`covar_mtx_mean_vec`](@ref), and [`_mu_cov_w`](@ref).

```math
\\begin{align*}
\\bm{\\Pi} &= \\begin{cases}
                    \\delta \\mathbf{\\Sigma} \\bm{w} &\\quad \\mathrm{if~ eq = true}\\\\
                      \\bm{\\mu} - r &\\quad \\mathrm{if~ eq = false}
                  \\end{cases}\\\\                            
\\mathbf{\\Omega} &= \\tau \\mathrm{Diagonal}\\left(\\mathbf{P} \\mathbf{\\Sigma} \\mathbf{P}^{\\intercal}\\right)\\\\
\\mathbf{M} &= \\left[ \\left(\\tau  \\mathbf{\\Sigma} \\right)^{-1} + \\mathbf{P}^{\\intercal} \\mathbf{\\Omega}^{-1} \\mathbf{P}\\right]^{-1}\\\\
\\bm{\\Pi}_{\\mathrm{BL}} &= \\mathbf{M} \\left[\\left(\\tau \\mathbf{\\Sigma}\\right)^{-1} \\bm{\\Pi} + \\mathbf{P}^{\\intercal} \\mathbf{\\Omega}^{-1} \\mathbf{Q} \\right]\\\\
\\tau &= \\dfrac{1}{T}\\\\
\\bm{\\mu}_{\\mathrm{BL}} &= \\bm{\\Pi}_{\\mathrm{BL}} + r\\\\
\\mathbf{\\Sigma}_{\\mathrm{BL}} &= \\mathbf{\\Sigma} + \\mathbf{M}
\\end{align*}
```

Where:

  - ``\\bm{\\Pi}``: is `N×1` the equilibrium excess returns, where `N` is the number of assets.
  - ``\\delta``: is the risk aversion parameter.
  - ``\\mathbf{\\Sigma}``: is the `N×N` asset covariance matrix, where `N` is the number of assets.
  - ``\\bm{w}``: is the `N×1` vector of benchmark asset weights, where `N` is the number of assets.
  - ``\\mathbf{P}``: is the `Nv×N` asset views matrix, where `Nv` is the number of asset views, and `N` the number of assets.
  - ``\\bm{Q}``: is the `Nv×1` asset views returns vector, where `Nv` is the number of asset views.
  - ``\\mathbf{\\Omega}``: is the `Nv×Nv` covariance matrix of the errors of the asset views, where `Nv` is the number of asset views.
  - ``\\mathbf{M}``: is an `N×N` intermediate covariance matrix, where `N` is the number of assets, and `M` the number of assets.
  - ``\\bm{\\Pi}_{\\mathbf{BL}}``: is the `N×1` equilibrium excess returns after being adjusted by the views, where `N` is the number of assets.
  - ``T``: is the number of returns observations.
  - ``\\bm{\\mu}_{\\mathbf{BL}}``: is the `N×1` vector of asset expected returns obtained via the Black-Litterman model, where `N` is the number of assets.
  - ``\\mathbf{\\Sigma}_{\\mathrm{BL}}``: is the `N×N` asset covariance matrix obtained via the Black-Litterman model, where `N` is the number of assets.

# Inputs

  - `returns`: `T×N` matrix of returns, where `T` is the number of returns observations, and `N` is the number of assets or factors.
  - `P`: `Nv×N` analyst's asset views matrix, can be relative or absolute, where `Nv` is the number of asset views, and `N` the number of assets.
  - `Q`: `Nv×1` analyst's asset viewsd returns vector, where `Nv` is the number of asset views.
  - `w`: `N×1` benchmark weights vector, where `N` is the number of assets.

## Options

  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the vanilla covariance matrix.
  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the vanilla expected returns vector.
  - `bl_opt`: instance of [`BLOpt`](@ref), defines the parameters for computing the Black-Litterman model's statistics.

# Outputs

  - `mu`: `N×1` Black-Litterman adjusted expected returns vector, where `N` is the number of assets.
  - `sigma`: `N×N` Black-Litterman adjusted covariance matrix, where `N` is the number of assets.
  - `w`: `N×1` Black-Litterman adjusted asset weights vector, where `N` is the number of assets.

!!! note

    Note that both `bl_opt`, and `mu_opt` have `rf` fields for the risk-free rate (see [`MuOpt`](@ref) and [`BLOpt`](@ref)). This gives users more granular control over the model.
"""
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
    mu, sigma, w, missing = _mu_cov_w(tau, omega, P, Pi, Q, rf, sigma, delta, T, N, bl_opt,
                                      :bl_cov)

    return mu, sigma, w
end

"""
```
bayesian_black_litterman(returns::AbstractMatrix, F::AbstractMatrix, B::AbstractMatrix,
                         P_f::AbstractMatrix, Q_f::AbstractVector;
                         cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;),
                         bl_opt::BLOpt = BLOpt(;))
```

Estimates the Bayesian Black-Litterman statistics according to [`BLFMMethods`](@ref). See [`covar_mtx_mean_vec`](@ref), and [`_omega`](@ref).

# Inputs

  - `returns`: `T×Na` matrix of asset returns, where `T` is the number of returns observations, and `Na` the number of assets.
  - `F`: `T×Nf` matrix of factor returns, where `T` is the number of returns observations, and `Nf` the number of factors.
  - `B`: is the `T×(Nf+c)` loadings matrix, where `T` is the number of returns observations, `Nf` the number of factors, and `c ∈ (0, 1)` represents the optional column for the constant term as described in [`FactorOpt`](@ref).
  - `P_F`: `Nvf×Nf` analyst's factor views matrix, can be relative or absolute, where `Nvf` is the number of factor views, and `Nf` the number of factors.
  - `Q_f`: `Nvf×1` analyst's factor views returns vector, where `Nvf` is the number of factor views.

## Options

  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the vanilla factor covariance matrix.
  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the vanilla factor expected returns vector.
  - `bl_opt`: instance of [`BLOpt`](@ref), defines the parameters for computing the factor Black-Litterman model's statistics.

# Outputs

  - `mu_b`: estimated asset expected returns vector via the Bayesian Black-Litterman model.
  - `cov_mtx_b`: estimated asset covariance via the Bayesian Black-Litterman model.
  - `w_b`: estimated benchmark weights via the Bayesian Black-Litterman model.

!!! note

    Note that both `bl_opt`, and `mu_opt` have `rf` fields for the risk-free rate (see [`MuOpt`](@ref) and [`BLOpt`](@ref)). This gives users more granular control over the model.
"""
function bayesian_black_litterman(returns::AbstractMatrix, F::AbstractMatrix,
                                  B::AbstractMatrix, P_f::AbstractMatrix,
                                  Q_f::AbstractVector; cov_opt::CovOpt = CovOpt(;),
                                  mu_opt::MuOpt = MuOpt(;), bl_opt::BLOpt = BLOpt(;))
    f_sigma, f_mu = covar_mtx_mean_vec(F; cov_opt = cov_opt, mu_opt = mu_opt)

    constant = bl_opt.constant
    error = bl_opt.error
    delta = bl_opt.delta
    rf = bl_opt.rf

    f_mu .-= rf

    if constant
        alpha = B[:, 1]
        B = B[:, 2:end]
    end

    tau = 1 / size(returns, 1)

    sigma = B * f_sigma * transpose(B)

    if error
        var_args = bl_opt.var_genfunc.args
        var_func = bl_opt.var_genfunc.func
        var_kwargs = bl_opt.var_genfunc.kwargs
        D = returns - F * transpose(B)
        D = Diagonal(vec(var_func(D, var_args...; var_kwargs...)))
        sigma .+= D
    end

    omega_f = _omega(P_f, tau * f_sigma)

    inv_sigma = sigma \ I
    inv_sigma_f = f_sigma \ I
    inv_omega_f = omega_f \ I
    sigma_hat = (inv_sigma_f + transpose(P_f) * inv_omega_f * P_f) \ I
    Pi_hat = sigma_hat * (inv_sigma_f * f_mu + transpose(P_f) * inv_omega_f * Q_f)
    inv_sigma_hat = sigma_hat \ I
    iish_b_is_b = (inv_sigma_hat + transpose(B) * inv_sigma * B) \ I
    is_b_iish_b_is_b = inv_sigma * B * iish_b_is_b

    sigma_bbl = (inv_sigma - is_b_iish_b_is_b * transpose(B) * inv_sigma) \ I
    Pi_bbl = sigma_bbl * is_b_iish_b_is_b * inv_sigma_hat * Pi_hat

    mu = Pi_bbl .+ rf

    if constant
        mu .+= alpha
    end

    w = ((delta * sigma_bbl) \ I) * mu

    return mu, sigma_bbl, w
end

"""
```
augmented_black_litterman(returns::AbstractMatrix, w::AbstractVector;
                          F::Union{AbstractMatrix, Nothing}   = nothing,
                          B::Union{AbstractMatrix, Nothing}   = nothing,
                          P::Union{AbstractMatrix, Nothing}   = nothing,
                          P_f::Union{AbstractMatrix, Nothing} = nothing,
                          Q::Union{AbstractVector, Nothing}   = nothing,
                          Q_f::Union{AbstractVector, Nothing} = nothing,
                          cov_opt::CovOpt                     = CovOpt(;),
                          mu_opt::MuOpt                       = MuOpt(;),
                          f_cov_opt::CovOpt                   = CovOpt(;),
                          f_mu_opt::MuOpt                     = MuOpt(;),
                          bl_opt::BLOpt                       = BLOpt(;))
```

Estimates the Augmented Black-Litterman statistics to [`BLFMMethods`](@ref). See [`covar_mtx_mean_vec`](@ref), [`_omega`](@ref), [`_Pi`](@ref), [`_mu_cov_w`](@ref), and [`_denoise_logo_mtx`](@ref).

# Inputs

  - `returns`: `T×Na` matrix of asset returns, where `T` is the number of returns observations, and `Na` the number of assets.
  - `w`: `Na×1` benchmark weights vector, where `Na` is the number of assets.
  - `F`: `T×Nf` matrix of factor returns, where `T` is the number of returns observations, and `Nf` the number of factors.
  - `B`: is the `T×(Nf+c)` loadings matrix, where `T` is the number of returns observations, `Nf` the number of factors, and `c ∈ (0, 1)` represents the optional column for the constant term as described in [`FactorOpt`](@ref).
  - `P`: `Nva×Na` analyst's asset views matrix, can be relative or absolute, where `Nva` is the number of asset views, and `Na` the number of assets.
  - `P_F`: `Nvf×Nf` analyst's factor views matrix, can be relative or absolute, where `Nvf` is the number of factor views, and `Nf` the number of factors.
  - `Q`: `Nva×1` analyst's asset views returns vector, where `Nva` is the number of asset views.
  - `Q_f`: `Nvf×1` analyst's factor views returns vector, where `Nvf` is the number of factor views.

The model can be adjusted depending on which views matrices and views expected returns vectors are provided.

  - If any of `P` or `Q` are provided, both must be provided.
  - If any of `P_f` or `Q_f` are provided, both must be provided, as well as both `B` and `F` must be provided.
  - If both `P` and `Q` are provided, but `P_f` and `Q_f` are not, the augmented model reduces to the traditional Black-Litterman model in [`black_litterman`](@ref). In other words, all augmented variables defined in [`BLFMMethods`](@ref) only include the asset statistics.
  - If `P_f` and `Q_f` are provided, but `P` and `Q` are not, the augmneted model computes its statistics based on the factor model only. In other words, all augmented variables defined in [`BLFMMethods`](@ref) only include the estimated asset statistics using the factor model.
  - If `P`, `Q`, `P_f` and `Q_f` are all provided, the full augmented model as defined in [`BLFMMethods`](@ref) is used.

## Options

  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the vanilla asset covariance matrix.
  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the vanilla asset expected returns vector.
  - `f_cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the vanilla factor covariance matrix.
  - `f_mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the vanilla factor expected returns vector.
  - `bl_opt`: instance of [`BLOpt`](@ref), defines the parameters for computing the factor Black-Litterman model's statistics.

# Outputs

  - `mu_a`: estimated asset expected returns vector via the Augmented Black-Litterman model.
  - `cov_mtx_a`: estimated asset covariance via the Augmented Black-Litterman model.
  - `w_a`: estimated benchmark weights via the Augmented Black-Litterman model.

!!! note

    Note that both `bl_opt`, `f_mu_opt`, and `mu_opt` have `rf` fields for the risk-free rate (see [`MuOpt`](@ref) and [`BLOpt`](@ref)). This gives users more granular control over the model.
"""
function augmented_black_litterman(returns::AbstractMatrix, w::AbstractVector;
                                   F::Union{AbstractMatrix, Nothing}   = nothing,
                                   B::Union{AbstractMatrix, Nothing}   = nothing,
                                   P::Union{AbstractMatrix, Nothing}   = nothing,
                                   P_f::Union{AbstractMatrix, Nothing} = nothing,
                                   Q::Union{AbstractVector, Nothing}   = nothing,
                                   Q_f::Union{AbstractVector, Nothing} = nothing,
                                   cov_opt::CovOpt                     = CovOpt(;),
                                   mu_opt::MuOpt                       = MuOpt(;),
                                   f_cov_opt::CovOpt                   = CovOpt(;),
                                   f_mu_opt::MuOpt                     = MuOpt(;),
                                   bl_opt::BLOpt                       = BLOpt(;))
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
        f_sigma, f_mu = covar_mtx_mean_vec(F; cov_opt = f_cov_opt, mu_opt = f_mu_opt)
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
        sigma_a = f_sigma
        P_a = P_f
        Q_a = Q_f
        omega_a = _omega(P_a, tau * sigma_a)
        Pi_a = _Pi(eq, delta, sigma_a * transpose(B), w, f_mu, rf)
    elseif all_asset_provided && all_factor_provided
        sigma_a = hcat(vcat(sigma, f_sigma * transpose(B)), vcat(B * f_sigma, f_sigma))

        zeros_1 = zeros(size(P_f, 1), size(P, 2))
        zeros_2 = zeros(size(P, 1), size(P_f, 2))

        P_a = hcat(vcat(P, zeros_1), vcat(zeros_2, P_f))
        Q_a = vcat(Q, Q_f)

        omega = _omega(P, tau * sigma)
        omega_f = _omega(P_f, tau * f_sigma)

        zeros_3 = zeros(size(omega, 1), size(omega_f, 1))

        omega_a = hcat(vcat(omega, transpose(zeros_3)), vcat(zeros_3, omega_f))

        Pi_a = _Pi(eq, delta, vcat(sigma, f_sigma * transpose(B)), w, vcat(mu, f_mu), rf)
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
```
black_litterman_statistics!(portfolio::Portfolio, P::AbstractMatrix, Q::AbstractVector;
                            w::AbstractVector = portfolio.bl_bench_weights,
                            cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;),
                            bl_opt::BLOpt = BLOpt(;))
```

Estimates the Black-Litterman statistics for a given `portfolio` in-place. See [`black_litterman`](@ref).

Modifies:

  - `portfolio.bl_mu`
  - `portfolio.bl_cov`
  - `portfolio.bl_bench_weights`

Depending on conditions, modifies:

  - `bl_opt.delta`

# Inputs

  - `portfolio`: instance of [`Portfolio`](@ref).

  - `P`: `Nv×N` analyst's views matrix, can be relative or absolute, where `Nv` is the number of views, and `N` the number of assets.
  - `Q`: `Nv×1` analyst's expected returns vector, where `Nv` is the number of views.
  - `w`: `N×1` benchmark weights vector, sets `portfolio.bl_bench_weights`, where `N` is the number of assets.

      + `isempty(w)`: every entry is assumed to be `1/N`.

## Options

  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the vanilla covariance matrix.

  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the vanilla expected returns vector.
  - `bl_opt`: instance of [`BLOpt`](@ref), defines the parameters for computing the Black-Litterman model's statistics.

      + `isnothing(bl_opt.delta)`: sets `bl_opt.delta` to:

        ```math
        \\delta = \\dfrac{\\bm{\\mu} \\cdot \\bm{w} - r}{\\bm{w}^{\\intercal} \\mathbf{\\Sigma} \\bm{w}}\\,.
        ```

        Where:

          * ``\\delta``: is `bl_opt.delta`.
          * ``\\bm{\\mu}``: is `portfolio.mu`.
          * ``\\bm{w}``: is `portfolio.bl_bench_weights`.
          * ``r``: is `bl_opt.rf`.
          * ``\\mathbf{\\Sigma}``: is `portfolio.cov`.

!!! note

    Note that both `bl_opt`, and `mu_opt` have `rf` fields for the risk-free rate (see [`MuOpt`](@ref) and [`BLOpt`](@ref)). This gives users more granular control over the model.
"""
function black_litterman_statistics!(portfolio::Portfolio, P::AbstractMatrix,
                                     Q::AbstractVector,
                                     w::AbstractVector = portfolio.bl_bench_weights;
                                     cov_opt::CovOpt = CovOpt(;), mu_opt::MuOpt = MuOpt(;),
                                     bl_opt::BLOpt = BLOpt(;))
    returns = portfolio.returns
    if isempty(w)
        w = fill(1 / size(portfolio.returns, 2), size(portfolio.returns, 2))
    end
    portfolio.bl_bench_weights = w

    if isnothing(bl_opt.delta)
        bl_opt.delta = (dot(portfolio.mu, w) - bl_opt.rf) / dot(w, portfolio.cov, w)
    end

    portfolio.bl_mu, portfolio.bl_cov, missing = black_litterman(returns, P, Q, w;
                                                                 cov_opt = cov_opt,
                                                                 mu_opt = mu_opt,
                                                                 bl_opt = bl_opt)

    return nothing
end

"""
```
factor_statistics!(portfolio::Portfolio; cov_opt::CovOpt = CovOpt(;),
                   mu_opt::MuOpt = MuOpt(;), factor_opt::FactorOpt = FactorOpt(;))
```

Compute the factor and factor adjusted statistics for a given `portfolio` in-place. See [`covar_mtx_mean_vec`](@ref), and [`risk_factors`](@ref).

Modifies:

  - `portfolio.f_mu`
  - `portfolio.f_cov`
  - `portfolio.fm_mu`
  - `portfolio.fm_cov`
  - `portfolio.fm_returns`
  - `portfolio.loadings`
  - `portfolio.loadings_opt`

# Inputs

  - `portfolio`: instance of [`Portfolio`](@ref).

## Options

  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the factor covariance matrix.
  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the factor expected returns vector.
  - `factor_opt`: instance of [`FactorOpt`](@ref), defines how the factor statistics are computed.
"""
function factor_statistics!(portfolio::Portfolio; cov_opt::CovOpt = CovOpt(;),
                            mu_opt::MuOpt = MuOpt(;), factor_opt::FactorOpt = FactorOpt(;))
    returns = portfolio.returns
    f_returns = portfolio.f_returns

    portfolio.f_cov, portfolio.f_mu = covar_mtx_mean_vec(f_returns; cov_opt = cov_opt,
                                                         mu_opt = mu_opt)

    portfolio.fm_mu, portfolio.fm_cov, portfolio.fm_returns, portfolio.loadings = risk_factors(DataFrame(f_returns,
                                                                                                         portfolio.f_assets),
                                                                                               DataFrame(returns,
                                                                                                         portfolio.assets);
                                                                                               factor_opt = factor_opt,
                                                                                               cov_opt = cov_opt,
                                                                                               mu_opt = mu_opt)

    portfolio.loadings_opt = factor_opt.loadings_opt

    return nothing
end

"""
```
black_litterman_factor_statistics!(portfolio::Portfolio,
                                  w::AbstractVector                   = portfolio.bl_bench_weights;
                                  B::Union{DataFrame, Nothing}        = portfolio.loadings,
                                  P::Union{AbstractMatrix, Nothing}   = nothing,
                                  P_f::Union{AbstractMatrix, Nothing} = nothing,
                                  Q::Union{AbstractVector, Nothing}   = nothing,
                                  Q_f::Union{AbstractVector, Nothing} = nothing,
                                  loadings_opt::LoadingsOpt           = LoadingsOpt(;),
                                  cov_opt::CovOpt                     = CovOpt(;),
                                  mu_opt::MuOpt                       = MuOpt(;),
                                  f_cov_opt::CovOpt                   = CovOpt(;),
                                  f_mu_opt::MuOpt                     = MuOpt(;),
                                  bl_opt::BLOpt                       = BLOpt(;))
```

Estimates the factor Black-Litterman statistics in-place according to [`BLFMMethods`](@ref). See [`augmented_black_litterman`](@ref), [`bayesian_black_litterman`](@ref).

Modifies:

  - `portfolio.blfm_mu`
  - `portfolio.blfm_cov`
  - `portfolio.bl_bench_weights`
  - `portfolio.loadings`
  - `bl_opt.constant`

Depending on conditions, modifies:

  - `bl_opt.delta`
  - `portfolio.loadings_opt`

# Inputs

  - `portfolio`: instance of [`Portfolio`](@ref).

  - `w`: `Na×1` benchmark weights vector, sets `portfolio.bl_bench_weights`, where `Na` is the number of assets.

      + `isempty(w)`: every entry is assumed to be `1/Na`.
  - `B`: is the `T×(Nf+c)` loadings matrix in Dataframe form, sets `portfolio.loadings` and `bl_opt.constant = "const" ∈ names(B)`, where `T` is the number of returns observations, `Nf` the number of factors, and `c ∈ (0, 1, 2)` represents the two optional columns as described in [`FactorOpt`](@ref).

      + `(isnothing(B) || isempty(B)) && isempty(portfolio.loadings)`: internally computes `B` using [`loadings_matrix`](@ref) and sets `portfolio.loadings_opt = loadings_opt`.
  - `P`: `Nva×Na` analyst's asset views matrix, can be relative or absolute, where `Nva` is the number of asset views, and `Na` the number of assets.
  - `P_F`: `Nvf×Nf` analyst's factor views matrix, can be relative or absolute, where `Nvf` is the number of factor views, and `Nf` the number of factors.
  - `Q`: `Nva×1` analyst's asset views returns vector, where `Nva` is the number of asset views.
  - `Q_f`: `Nvf×1` analyst's factor views returns vector, where `Nvf` is the number of factor views.

## Options

  - `loadings_opt`: instance of [`LoadingsOpt`](@ref), defines the parameters for computing the loadings matrix.

  - `cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the vanilla asset covariance matrix.
  - `mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the vanilla asset expected returns vector.
  - `f_cov_opt`: instance of [`CovOpt`](@ref), defines the parameters for computing the vanilla factor covariance matrix.
  - `f_mu_opt`: instance of [`MuOpt`](@ref), defines the parameters for computing the vanilla factor expected returns vector.
  - `bl_opt`: instance of [`BLOpt`](@ref), defines the parameters for computing the factor Black-Litterman model's statistics.

      + `isnothing(bl_opt.delta)`: sets `bl_opt.delta` to:

        ```math
        \\delta = \\dfrac{\\bm{\\mu} \\cdot \\bm{w} - r}{\\bm{w}^{\\intercal} \\mathbf{\\Sigma} \\bm{w}}\\,.
        ```

        Where:

          * ``\\delta``: is `bl_opt.delta`.
          * ``\\bm{\\mu}``: is `portfolio.mu`.
          * ``\\bm{w}``: is `portfolio.bl_bench_weights`.
          * ``r``: is `bl_opt.rf`.
          * ``\\mathbf{\\Sigma}``: is `portfolio.cov`.

!!! note

    Note that both `bl_opt`, `f_mu_opt`, and `mu_opt` have `rf` fields for the risk-free rate (see [`MuOpt`](@ref) and [`BLOpt`](@ref)). This gives users more granular control over the model.
"""
function black_litterman_factor_statistics!(portfolio::Portfolio,
                                            w::AbstractVector                   = portfolio.bl_bench_weights;
                                            B::Union{DataFrame, Nothing}        = portfolio.loadings,
                                            P::Union{AbstractMatrix, Nothing}   = nothing,
                                            P_f::Union{AbstractMatrix, Nothing} = nothing,
                                            Q::Union{AbstractVector, Nothing}   = nothing,
                                            Q_f::Union{AbstractVector, Nothing} = nothing,
                                            loadings_opt::LoadingsOpt           = LoadingsOpt(;),
                                            cov_opt::CovOpt                     = CovOpt(;),
                                            mu_opt::MuOpt                       = MuOpt(;),
                                            f_cov_opt::CovOpt                   = CovOpt(;),
                                            f_mu_opt::MuOpt                     = MuOpt(;),
                                            bl_opt::BLOpt                       = BLOpt(;))
    returns = portfolio.returns
    F = portfolio.f_returns

    if isempty(w)
        w = fill(1 / size(portfolio.returns, 2), size(portfolio.returns, 2))
    end
    portfolio.bl_bench_weights = w

    if isnothing(bl_opt.delta)
        bl_opt.delta = (dot(portfolio.mu, w) - bl_opt.rf) / dot(w, portfolio.cov, w)
    end

    if isnothing(B) || isempty(B)
        if isempty(portfolio.loadings)
            portfolio.loadings = loadings_matrix(DataFrame(F, portfolio.f_assets),
                                                 DataFrame(returns, portfolio.assets),
                                                 loadings_opt)
            portfolio.loadings_opt = loadings_opt
        end
        B = portfolio.loadings
    else
        portfolio.loadings = B
    end

    namesB = names(B)
    bl_opt.constant = "const" ∈ namesB
    B = Matrix(B[!, setdiff(namesB, ("tickers",))])

    portfolio.blfm_mu, portfolio.blfm_cov, missing = if bl_opt.method == :B
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

"""
```
_hierarchical_clustering
```
"""
function _hierarchical_clustering(returns::AbstractMatrix, cor_opt::CorOpt = CorOpt(;),
                                  cluster_opt::ClusterOpt = ClusterOpt(;))
    corr, dist = cor_dist_mtx(returns, cor_opt)
    clustering, k = _hcluster_choice(corr, dist, cluster_opt)

    return clustering, k, corr, dist
end

"""
```
cluster_assets
```
"""
function cluster_assets(portfolio::HCPortfolio, opt::ClusterOpt = ClusterOpt(;))
    clustering, tk = _hierarchical_clustering(portfolio, opt)

    k = iszero(opt.k) ? tk : opt.k

    clustering_idx = cutree(clustering; k = k)

    return clustering_idx, clustering, k
end

"""
```
cluster_assets!
```
"""
function cluster_assets!(portfolio::HCPortfolio, opt::ClusterOpt = ClusterOpt(;))
    clustering, tk = _hierarchical_clustering(portfolio, opt)

    k = iszero(opt.k) ? tk : opt.k

    portfolio.clusters = clustering
    portfolio.k = k

    return nothing
end

"""
```
cluster_assets
```
"""
function cluster_assets(returns::AbstractMatrix; cor_opt::CorOpt = CorOpt(;),
                        cluster_opt::ClusterOpt = ClusterOpt(;))
    clustering, tk, corr, dist = _hierarchical_clustering(returns, cor_opt, cluster_opt)

    k = iszero(cluster_opt.k) ? tk : cluster_opt.k

    clustering_idx = cutree(clustering; k = k)

    return clustering_idx, clustering, k, corr, dist
end

"""
```
cluster_assets
```
"""
function cluster_assets(portfolio::Portfolio; cor_opt::CorOpt = CorOpt(;),
                        cluster_opt::ClusterOpt = ClusterOpt(;))
    return cluster_assets(portfolio.returns; cor_opt = cor_opt, cluster_opt = cluster_opt)
end

export asset_statistics!, wc_statistics!, loadings_matrix, black_litterman_statistics!,
       factor_statistics!, black_litterman_factor_statistics!, cluster_assets,
       cluster_assets!

####################

# # Auxiliary Functions
# ## Fix matrices that should be positive definite.
abstract type PosdefFix end
struct NoPosdef <: PosdefFix end
@kwdef mutable struct PosdefNearest <: PosdefFix
    method::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                    tau = 1e-12)
end
"""
```
_posdef_fix!(method::PosdefNearest, X::AbstractMatrix)
```

Overload this for other posdef fix methods.
"""
function _posdef_fix!(method::PosdefNearest, X::AbstractMatrix)
    NCM.nearest_cor!(X, method)
    return nothing
end
function posdef_fix!(::NoPosdef, ::AbstractMatrix)
    return nothing
end
function posdef_fix!(method::PosdefFix, X::AbstractMatrix)
    if isposdef(X)
        return nothing
    end

    s = diag(X)
    iscov = any(.!isone.(s))
    _X = if iscov
        s .= sqrt.(s)
        cov2cor(X, s)
    else
        X
    end

    _posdef_fix!(method, _X)

    if !isposdef(_X)
        @warn("Matrix could not be made positive definite.")
        return nothing
    end

    if iscov
        StatsBase.cor2cov!(_X, s)
    end

    X .= _X

    return nothing
end

# # DBHT Similarity Matrices
abstract type DBHTSimilarity end
struct DBHTExp <: DBHTSimilarity end
function dbht_similarity(::DBHTExp, S, D)
    return exp.(-D)
end
struct DBHTMaxDist <: DBHTSimilarity end
function dbht_similarity(::DBHTMaxDist, S, D)
    return ceil(maximum(D)^2) .- D .^ 2
end

# # Distance matrices
abstract type DistanceMethod end
@kwdef mutable struct DistanceMLP <: DistanceMethod
    absolute::Bool = false
end
function _dist(de::DistanceMLP, X::AbstractMatrix, ::Any)
    return Symmetric(sqrt.(if !de.absolute
                               clamp!((one(eltype(X)) .- X) / 2, zero(eltype(X)),
                                      one(eltype(X)))
                           else
                               clamp!(one(eltype(X)) .- X, zero(eltype(X)), one(eltype(X)))
                           end))
end
@kwdef mutable struct DistanceMLP2 <: DistanceMethod
    absolute::Bool = false
    distance::Distances.UnionMetric = Distances.Euclidean()
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
function _dist(de::DistanceMLP2, X::AbstractMatrix, ::Any)
    _X = sqrt.(if !de.absolute
                   clamp!((one(eltype(X)) .- X) / 2, zero(eltype(X)), one(eltype(X)))
               else
                   clamp!(one(eltype(X)) .- X, zero(eltype(X)), one(eltype(X)))
               end)

    return Distances.pairwise(de.distance, _X, de.::Any; de.kwargs...)
end
struct DistanceLog <: DistanceMethod end
function _dist(::DistanceLog, X::AbstractMatrix, ::Any)
    return -log.(X)
end
function dist(de::DistanceMethod, X, Y)
    return _dist(de, X, Y)
end
struct DistanceDefault <: DistanceMethod end

@kwdef mutable struct SimpleVariance <: StatsBase.CovarianceEstimator
    corrected = true
end
function StatsBase.std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return std(X; dims = dims, corrected = ve.corrected, mean = mean)
end
function StatsBase.std(ve::SimpleVariance, X::AbstractMatrix, w::AbstractWeights;
                       dims::Int = 1, mean = nothing)
    return std(X, w, dims; corrected = ve.corrected, mean = mean)
end

# # Correlation Matrices
abstract type PortfolioOptimiserCovCor <: StatsBase.CovarianceEstimator end
abstract type CorPearson <: PortfolioOptimiserCovCor end
@kwdef mutable struct CovFull <: CorPearson
    absolute::Bool = false
    ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    w::Union{<:AbstractWeights, Nothing} = nothing
end
function StatsBase.cov(ce::CovFull, X::AbstractMatrix; dims::Int = 1)
    return Symmetric(if isnothing(ce.w)
                         cov(ce.ce, X; dims = dims)
                     else
                         cov(ce.ce, X, ce.w; dims = dims)
                     end)
end
function StatsBase.cor(ce::CovFull, X::AbstractMatrix; dims::Int = 1)
    rho = Symmetric(try
                        if isnothing(ce.w)
                            cor(ce.ce, X; dims = dims)
                        else
                            cor(ce.ce, X, ce.w; dims = dims)
                        end
                    catch
                        StatsBase.cov2cor(Matrix(if isnothing(ce.w)
                                                     cov(ce.ce, X; dims = dims)
                                                 else
                                                     cov(ce.ce, X, ce.w; dims = dims)
                                                 end))
                    end)

    return !ce.absolute ? rho : abs.(rho)
end
@kwdef mutable struct CovSemi <: CorPearson
    absolute::Bool = false
    ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)
    target::Union{Real, AbstractVector{<:Real}} = 0.0
    w::Union{<:AbstractWeights, Nothing} = nothing
end
function StatsBase.cov(ce::CovSemi, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (0, 1))
    if dims == 2
        X = transpose(X)
    end
    target = ce.target
    X = if isa(target, Real)
        min.(X .- target, zero(eltype(X)))
    else
        min.(X .- transpose(target), zero(eltype(X)))
    end
    return Symmetric(if isnothing(ce.w)
                         cov(ce.ce, X; mean = zero(eltype(X)))
                     else
                         cov(ce.ce, X, ce.w; mean = zero(eltype(X)))
                     end)
end
function StatsBase.cor(ce::CovSemi, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (0, 1))
    if dims == 2
        X = transpose(X)
    end
    target = ce.target
    X = if isa(target, Real)
        min.(X .- target, zero(eltype(X)))
    else
        min.(X .- transpose(target), zero(eltype(X)))
    end

    rho = Symmetric(try
                        if isnothing(ce.w)
                            cor(ce.ce, X; mean = zero(eltype(X)))
                        else
                            cor(ce.ce, X, ce.w; mean = zero(eltype(X)))
                        end
                    catch
                        StatsBase.cov2cor(Matrix(if isnothing(ce.w)
                                                     cov(ce.ce, X; mean = zero(eltype(X)))
                                                 else
                                                     cov(ce.ce, X, ce.w;
                                                         mean = zero(eltype(X)))
                                                 end))
                    end)
    return !ce.absolute ? rho : abs.(rho)
end
function StatsBase.cor(ce::CorPearson, X; dims::Int = 1)
    return cor(ce, X; dims = dims)
end
@kwdef mutable struct CorSpearman <: PortfolioOptimiserCovCor
    absolute::Bool = false
end
function StatsBase.cor(ce::CorSpearman, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (0, 1))
    if dims == 2
        X = transpose(X)
    end
    rho = corspearman(X)
    return Symmetric(cov2cor(Matrix(!ce.absolute ? rho : abs.(rho))))
end
@kwdef mutable struct CorKendall <: PortfolioOptimiserCovCor
    absolute::Bool = false
end
function StatsBase.cor(ce::CorKendall, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (0, 1))
    if dims == 2
        X = transpose(X)
    end
    rho = corkendall(X)
    return Symmetric(cov2cor(Matrix(!ce.absolute ? rho : abs.(rho))))
end

# ## Mutual and variation of information matrices.
abstract type AbstractBins end
abstract type AstroBins <: AbstractBins end
struct BinKnuth <: AstroBins end
struct BinFreedman <: AstroBins end
struct BinScott <: AstroBins end
struct BinHGR <: AbstractBins end

mutable struct DistanceVarInfo <: DistanceMethod
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
end
function DistanceVarInfo(; bins::Union{<:Integer, <:AbstractBins} = BinHGR(),
                         normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return DistanceVarInfo(bins, normalise)
end
function Base.setproperty!(obj::DistanceVarInfo, sym::Symbol, val)
    if sym == :bins
        if isa(val, Integer)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end
mutable struct CorMutualInfo <: PortfolioOptimiserCovCor
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
end
function CorMutualInfo(; bins::Union{<:Integer, <:AbstractBins} = BinHGR(),
                       normalise::Bool = true,
                       ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                       std_w::Union{<:AbstractWeights, Nothing} = nothing)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return CorMutualInfo(bins, normalise, ve, std_w)
end
function Base.setproperty!(obj::CorMutualInfo, sym::Symbol, val)
    if sym == :bins
        if isa(val, Integer)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end
function _bin_width_func(::BinKnuth)
    return pyimport("astropy.stats").knuth_bin_width
end
function _bin_width_func(::BinFreedman)
    return pyimport("astropy.stats").freedman_bin_width
end
function _bin_width_func(::BinScott)
    return pyimport("astropy.stats").scott_bin_width
end
function _bin_width_func(::Union{BinHGR, <:Integer})
    return nothing
end
function calc_num_bins(::AstroBins, xj, xi, j::Integer, i, bin_width_func, T = nothing)
    k1 = (maximum(xj) - minimum(xj)) / bin_width_func(xj)
    return round(Int, if j != i
                     k2 = (maximum(xi) - minimum(xi)) / bin_width_func(xi)
                     max(k1, k2)
                 else
                     k1
                 end)
end
function calc_num_bins(::BinHGR, xj, xi, j, i, bin_width_func, N)
    corr = cor(xj, xi)
    return round(Int, if isone(corr)
                     z = cbrt(8 + 324 * N + 12 * sqrt(36 * N + 729 * N^2))
                     z / 6 + 2 / (3 * z) + 1 / 3
                 else
                     sqrt(1 + sqrt(1 + 24 * N / (1 - corr^2))) / sqrt(2)
                 end)
end
function calc_num_bins(bins::Integer, xj, xi, j, i, bin_width_func, N)
    return bins
end
function calc_hist_data(xj::AbstractVector, xi::AbstractVector, bins::Integer)
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
function _mutual_info(A::AbstractMatrix)
    p_i = vec(sum(A; dims = 2))
    p_j = vec(sum(A; dims = 1))

    if length(p_i) == 1 || length(p_j) == 1
        return zero(eltype(p_j))
    end

    mask = findall(.!iszero.(A))

    nz = vec(A[mask])
    nz_sum = sum(nz)
    log_nz = log.(nz)
    nz_nm = nz / nz_sum

    outer = p_i[getindex.(mask, 1)] .* p_j[getindex.(mask, 2)]
    log_outer = -log.(outer) .+ log(sum(p_i)) .+ log(sum(p_j))

    mi = (nz_nm .* (log_nz .- log(nz_sum)) .+ nz_nm .* log_outer)
    mi[abs.(mi) .< eps(eltype(mi))] .= zero(eltype(A))

    return sum(mi)
end
function mutual_variation_info(X::AbstractMatrix,
                               bins::Union{<:AbstractBins, <:Integer} = BinKnuth(),
                               normalise::Bool = true)
    T, N = size(X)
    mut_mtx = Matrix{eltype(X)}(undef, N, N)
    var_mtx = Matrix{eltype(X)}(undef, N, N)

    bin_width_func = _bin_width_func(bins)

    for j ∈ eachindex(axes(X, 2))
        xj = X[:, j]
        for i ∈ 1:j
            xi = X[:, i]
            nbins = calc_num_bins(bins, xj, xi, j, i, bin_width_func, T)
            ex, ey, hxy = calc_hist_data(xj, xi, nbins)

            mut_ixy = _mutual_info(hxy)
            var_ixy = ex + ey - 2 * mut_ixy
            if normalise
                vxy = ex + ey - mut_ixy
                var_ixy = var_ixy / vxy
                mut_ixy /= min(ex, ey)
            end

            if abs(mut_ixy) < eps(typeof(mut_ixy)) || mut_ixy < zero(eltype(X))
                mut_ixy = zero(eltype(X))
            end
            if abs(var_ixy) < eps(typeof(var_ixy)) || var_ixy < zero(eltype(X))
                var_ixy = zero(eltype(X))
            end

            mut_mtx[i, j] = mut_ixy
            var_mtx[i, j] = var_ixy
        end
    end

    return Symmetric(mut_mtx, :U), Symmetric(var_mtx, :U)
end
function mutual_info(X::AbstractMatrix, bins::Union{<:AbstractBins, <:Integer} = BinHGR(),
                     normalise::Bool = true)
    T, N = size(X)
    mut_mtx = Matrix{eltype(X)}(undef, N, N)

    bin_width_func = _bin_width_func(bins)

    for j ∈ eachindex(axes(X, 2))
        xj = X[:, j]
        for i ∈ 1:j
            xi = X[:, i]
            nbins = calc_num_bins(bins, xj, xi, j, i, bin_width_func, T)
            ex, ey, hxy = calc_hist_data(xj, xi, nbins)

            mut_ixy = _mutual_info(hxy)
            if normalise
                vxy = ex + ey - mut_ixy
                mut_ixy /= min(ex, ey)
            end

            if abs(mut_ixy) < eps(typeof(mut_ixy)) || mut_ixy < zero(eltype(X))
                mut_ixy = zero(eltype(X))
            end

            mut_mtx[i, j] = mut_ixy
        end
    end

    return Symmetric(mut_mtx, :U)
end
function variation_info(X::AbstractMatrix,
                        bins::Union{<:AbstractBins, <:Integer} = BinHGR(),
                        normalise::Bool = true)
    T, N = size(X)
    var_mtx = Matrix{eltype(X)}(undef, N, N)

    bin_width_func = _bin_width_func(bins)

    for j ∈ eachindex(axes(X, 2))
        xj = X[:, j]
        for i ∈ 1:j
            xi = X[:, i]
            nbins = calc_num_bins(bins, xj, xi, j, i, bin_width_func, T)
            ex, ey, hxy = calc_hist_data(xj, xi, nbins)

            mut_ixy = _mutual_info(hxy)
            var_ixy = ex + ey - 2 * mut_ixy
            if normalise
                vxy = ex + ey - mut_ixy
                var_ixy = var_ixy / vxy
            end

            if abs(var_ixy) < eps(typeof(var_ixy)) || var_ixy < zero(eltype(X))
                var_ixy = zero(eltype(X))
            end

            var_mtx[i, j] = var_ixy
        end
    end

    return Symmetric(var_mtx, :U)
end
function StatsBase.cor(ce::CorMutualInfo, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (0, 1))
    if dims == 2
        X = transpose(X)
    end
    return mutual_info(X, ce.bins, ce.normalise)
end
function StatsBase.cov(ce::CorMutualInfo, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (0, 1))
    if dims == 2
        X = transpose(X)
    end
    std_vec = vec(if isnothing(ce.std_w)
                      std(ce.ve, X; dims = 1)
                  else
                      std(ce.ve, X, ce.std_w; dims = 1)
                  end)
    return Symmetric(mutual_info(X, ce.bins, ce.normalise) .*
                     (std_vec * transpose(std_vec)))
end
function _dist(ce::DistanceVarInfo, ::Any, Y::AbstractMatrix)
    return variation_info(Y, ce.bins, ce.normalise)
end

# ## Distance correlation.
@kwdef mutable struct CorDistance <: PortfolioOptimiserCovCor
    distance::Distances.UnionMetric = Distances.Euclidean()
    dist_args::Tuple = ()
    dist_kwargs::NamedTuple = (;)
    mean_w1::Union{<:AbstractWeights, Nothing} = nothing
    mean_w2::Union{<:AbstractWeights, Nothing} = nothing
    mean_w3::Union{<:AbstractWeights, Nothing} = nothing
end
function cor_distance(ce::CorDistance, v1::AbstractVector, v2::AbstractVector)
    N = length(v1)
    @smart_assert(N == length(v2) && N > 1)

    N2 = N^2

    a = Distances.pairwise(ce.distance, v1, ce.dist_args...; ce.dist_kwargs...)
    b = Distances.pairwise(ce.distance, v2, ce.dist_args...; ce.dist_kwargs...)

    mu_a1, mu_b1 = if isnothing(ce.mean_w1)
        mean(a; dims = 1), mean(b; dims = 1)
    else
        mean(a, ce.mean_w1; dims = 1), mean(b, ce.mean_w1; dims = 1)
    end
    mu_a2, mu_b2 = if isnothing(ce.mean_w2)
        mean(a; dims = 2), mean(b; dims = 2)
    else
        mean(a, ce.mean_w2; dims = 2), mean(b, ce.mean_w2; dims = 2)
    end
    mu_a3, mu_b3 = if isnothing(ce.mean_w3)
        mean(a), mean(b)
    else
        mean(a, ce.mean_w3), mean(b, ce.mean_w3)
    end

    A = a .- mu_a1 .- mu_a2 .+ mu_a3
    B = b .- mu_b1 .- mu_b2 .+ mu_b3

    dcov2_xx = sum(A .* A) / N2
    dcov2_xy = sum(A .* B) / N2
    dcov2_yy = sum(B .* B) / N2

    val = sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))

    return val
end
function cor_distance(ce::CorDistance, X::AbstractMatrix)
    N = size(X, 2)

    rho = Matrix{eltype(X)}(undef, N, N)
    for j ∈ eachindex(axes(X, 2))
        xj = X[:, j]
        for i ∈ 1:j
            rho[i, j] = cor_distance(ce, X[:, i], xj)
        end
    end

    return Symmetric(rho, :U)
end
function StatsBase.cor(ce::CorDistance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (0, 1))
    if dims == 2
        X = transpose(X)
    end
    return cor_distance(ce::CorDistance, X::AbstractMatrix)
end
function cov_distance(ce::CorDistance, v1::AbstractVector, v2::AbstractVector)
    N = length(v1)
    @smart_assert(N == length(v2) && N > 1)

    N2 = N^2

    a = Distances.pairwise(ce.distance, v1, ce.dist_args...; ce.dist_kwargs...)
    b = Distances.pairwise(ce.distance, v2, ce.dist_args...; ce.dist_kwargs...)

    mu_a1, mu_b1 = if isnothing(ce.mean_w1)
        mean(a; dims = 1), mean(b; dims = 1)
    else
        mean(a, ce.mean_w1; dims = 1), mean(b, ce.mean_w1; dims = 1)
    end
    mu_a2, mu_b2 = if isnothing(ce.mean_w2)
        mean(a; dims = 2), mean(b; dims = 2)
    else
        mean(a, ce.mean_w2; dims = 2), mean(b, ce.mean_w2; dims = 2)
    end
    mu_a3, mu_b3 = if isnothing(ce.mean_w3)
        mean(a), mean(b)
    else
        mean(a, ce.mean_w3), mean(b, ce.mean_w3)
    end

    A = a .- mu_a1 .- mu_a2 .+ mu_a3
    B = b .- mu_b1 .- mu_b2 .+ mu_b3

    dcov2_xy = sum(A .* B) / N2

    val = sqrt(dcov2_xy)

    return val
end
function cov_distance(ce::CorDistance, X::AbstractMatrix)
    N = size(X, 2)

    rho = Matrix{eltype(X)}(undef, N, N)
    for j ∈ eachindex(axes(X, 2))
        xj = X[:, j]
        for i ∈ 1:j
            rho[i, j] = cov_distance(ce, X[:, i], xj)
        end
    end

    return Symmetric(rho, :U)
end
function StatsBase.cov(ce::CorDistance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (0, 1))
    if dims == 2
        X = transpose(X)
    end
    return Symmetric(cov_distance(ce, X))
end

# ## Lower tail correlation.
mutable struct CorLTD <: PortfolioOptimiserCovCor
    alpha::Real
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
end
function CorLTD(; alpha::Real = 0.05, ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                std_w::Union{<:AbstractWeights, Nothing} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CorLTD(alpha, ve, std_w)
end
function Base.setproperty!(obj::CorLTD, sym::Symbol, val)
    if sym == :alpha
        @smart_assert(zero(val) <= val <= one(val))
    end
    return setfield!(obj, sym, val)
end
function lower_tail_dependence(X::AbstractMatrix, alpha::Real = 0.05)
    T, N = size(X)
    k = ceil(Int, T * alpha)
    rho = Matrix{eltype(X)}(undef, N, N)

    if k > 0
        for j ∈ eachindex(axes(X, 2))
            xj = X[:, j]
            v = sort(xj)[k]
            maskj = xj .<= v
            for i ∈ 1:j
                xi = X[:, i]
                u = sort(xi)[k]
                ltd = sum(xi .<= u .&& maskj) / k
                rho[i, j] = clamp(ltd, zero(eltype(X)), one(eltype(X)))
            end
        end
    end

    return Symmetric(rho, :U)
end
function StatsBase.cor(ce::CorLTD, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (0, 1))
    if dims == 2
        X = transpose(X)
    end
    return lower_tail_dependence(X, ce.alpha)
end
function StatsBase.cov(ce::CorLTD, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (0, 1))
    if dims == 2
        X = transpose(X)
    end
    std_vec = vec(if isnothing(ce.std_w)
                      std(ce.ve, X; dims = 1)
                  else
                      std(ce.ve, X, ce.std_w; dims = 1)
                  end)
    return lower_tail_dependence(X, ce.alpha) .* (std_vec * transpose(std_vec))
end

# ## Gerber covariances
abstract type CorGerber <: PortfolioOptimiserCovCor end
abstract type CorGerberBasic <: CorGerber end
abstract type CorSB <: CorGerber end
abstract type CorGerberSB <: CorGerber end
mutable struct CorGerber0{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerber0(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::PosdefFix = PosdefNearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber0{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end
mutable struct CorGerber1{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerber1(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::PosdefFix = PosdefNearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber1{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end
mutable struct CorGerber2{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerber2(; normalise::Bool = false, threshold::Real = 0.5,
                    ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                    std_w::Union{<:AbstractWeights, Nothing} = nothing,
                    mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                    posdef::PosdefFix = PosdefNearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber2{typeof(threshold)}(normalise, threshold, ve, std_w, mean_w, posdef)
end
function Base.setproperty!(obj::CorGerberBasic, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end
function _cor_gerber_norm(ce::CorGerber0, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold

    for j ∈ eachindex(axes(X, 2))
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = 0
            pos = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (X[k, i] - mui) / sigmai
                xj = (X[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += 1
                end
            end
            den = (pos + neg)
            rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber(ce::CorGerber0, X::AbstractMatrix, std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold

    for j ∈ eachindex(axes(X, 2))
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = 0
            pos = 0
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += 1
                end
            end
            den = (pos + neg)
            rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber_norm(ce::CorGerber1, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold

    for j ∈ eachindex(axes(X, 2))
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = 0
            pos = 0
            nn = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (X[k, i] - mui) / sigmai
                xj = (X[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += 1
                end
            end
            den = (T - nn)
            rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber(ce::CorGerber1, X::AbstractMatrix, std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold

    for j ∈ eachindex(axes(X, 2))
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = 0
            pos = 0
            nn = 0
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += 1
                end
            end
            den = (T - nn)
            rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber_norm(ce::CorGerber2, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    threshold = ce.threshold

    @inbounds for i ∈ eachindex(axes(X, 2))
        xi = (X[:, i] .- mean_vec[i]) / std_vec[i]
        ti = threshold
        U[:, i] .= xi .>= ti
        D[:, i] .= xi .<= -ti
    end

    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc

    UmD = U - D
    H = transpose(UmD) * (UmD)

    h = sqrt.(diag(H))

    rho = H ./ (h * transpose(h))
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber(ce::CorGerber2, X::AbstractMatrix, std_vec::AbstractVector)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    threshold = ce.threshold

    @inbounds for i ∈ 1:N
        xi = X[:, i]
        ti = threshold * std_vec[i]
        U[:, i] .= xi .>= ti
        D[:, i] .= xi .<= -ti
    end

    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc

    UmD = U - D
    H = transpose(UmD) * (UmD)

    h = sqrt.(diag(H))

    rho = H ./ (h * transpose(h))
    posdef_fix!(ce.posdef, rho)

    return rho
end
mutable struct CorSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorSB0(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4, n::Real = 2,
                ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                std_w::Union{<:AbstractWeights, Nothing} = nothing,
                mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                posdef::PosdefFix = PosdefNearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorSB0{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                    threshold,
                                                                                    c1, c2,
                                                                                    c3, n,
                                                                                    ve,
                                                                                    std_w,
                                                                                    mean_w,
                                                                                    posdef)
end
mutable struct CorSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorSB1(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4, n::Real = 2,
                ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                std_w::Union{<:AbstractWeights, Nothing} = nothing,
                mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                posdef::PosdefFix = PosdefNearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorSB0{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                    threshold,
                                                                                    c1, c2,
                                                                                    c3, n,
                                                                                    ve,
                                                                                    std_w,
                                                                                    mean_w,
                                                                                    posdef)
end
mutable struct CorGerberSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerberSB0(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4, n::Real = 2,
                      ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                      std_w::Union{<:AbstractWeights, Nothing} = nothing,
                      mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                      posdef::PosdefFix = PosdefNearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorGerberSB0{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                          threshold,
                                                                                          c1,
                                                                                          c2,
                                                                                          c3,
                                                                                          n,
                                                                                          ve,
                                                                                          std_w,
                                                                                          mean_w,
                                                                                          posdef)
end
mutable struct CorGerberSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    ve::StatsBase.CovarianceEstimator
    std_w::Union{<:AbstractWeights, Nothing}
    mean_w::Union{<:AbstractWeights, Nothing}
    posdef::PosdefFix
end
function CorGerberSB1(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4, n::Real = 2,
                      ve::StatsBase.CovarianceEstimator = SimpleVariance(;),
                      std_w::Union{<:AbstractWeights, Nothing} = nothing,
                      mean_w::Union{<:AbstractWeights, Nothing} = nothing,
                      posdef::PosdefFix = PosdefNearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorGerberSB1{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                          threshold,
                                                                                          c1,
                                                                                          c2,
                                                                                          c3,
                                                                                          n,
                                                                                          ve,
                                                                                          std_w,
                                                                                          mean_w,
                                                                                          posdef)
end
function Base.setproperty!(obj::CorSB, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    elseif sym ∈ (:c1, :c2)
        @smart_assert(zero(val) < val <= one(val))
    elseif sym == :c3
        @smart_assert(val > obj.c2)
    end
    return setfield!(obj, sym, val)
end
#=
function _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
    # Zone of confusion.
    # If the return is not a significant proportion of the standard deviation, we classify it as noise.
    if abs(xi) < sigmai * c1 && abs(xj) < sigmaj * c1
        return zero(eltype(xi))
    end

    # Zone of indecision.
    # Center returns at mu = 0 and rho = 1.
    ri = abs((xi - mui) / sigmai)
    rj = abs((xj - muj) / sigmaj)
    # If the return is less than c2 standard deviations, or greater than c3 standard deviations, we can't make a call since it may be noise, or overall market forces.
    if ri < c2 && rj < c2 || ri > c3 && rj > c3
        return zero(eltype(xi))
    end

    kappa = sqrt((1 + ri) * (1 + rj))
    gamma = abs(ri - rj)

    return kappa / (1 + gamma^n)
end
=#
function _cor_gerber_norm(ce::CorSB0, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    for j ∈ eachindex(axes(X, 2))
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (X[k, i] - mui) / sigmai
                xj = (X[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                     one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                     one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                end
            end
            den = (pos + neg)
            rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber(ce::CorSB0, X::AbstractMatrix, mean_vec::AbstractVector,
                     std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    for j ∈ eachindex(axes(X, 2))
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                end
            end
            den = (pos + neg)
            rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber_norm(ce::CorSB1, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    for j ∈ eachindex(axes(X, 2))
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (X[k, i] - mui) / sigmai
                xj = (X[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                     one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                     one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += _sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                end
            end
            den = (pos + neg + nn)
            rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber(ce::CorSB1, X::AbstractMatrix, mean_vec::AbstractVector,
                     std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    for j ∈ eachindex(axes(X, 2))
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                end
            end
            den = (pos + neg + nn)
            rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber_norm(ce::CorGerberSB0, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    for j ∈ eachindex(axes(X, 2))
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (X[k, i] - mui) / sigmai
                xj = (X[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                     one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                     one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cneg += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            den = (tpos + tneg)
            rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber(ce::CorGerberSB0, X::AbstractMatrix, mean_vec::AbstractVector,
                     std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    for j ∈ eachindex(axes(X, 2))
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            den = (tpos + tneg)
            rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber_norm(ce::CorGerberSB1, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    for j ∈ eachindex(axes(X, 2))
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            cneg = 0
            cpos = 0
            cnn = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = (X[k, i] - mui) / sigmai
                xj = (X[k, j] - muj) / sigmaj
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                     one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                     one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cneg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += _sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cnn += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            tnn = nn * cnn
            den = (tpos + tneg + tnn)
            rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _cor_gerber(ce::CorGerberSB1, X::AbstractMatrix, mean_vec::AbstractVector,
                     std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    for j ∈ eachindex(axes(X, 2))
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            cneg = 0
            cpos = 0
            cnn = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += _sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cnn += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            tnn = nn * cnn
            den = (tpos + tneg + tnn)
            rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end

    rho .= Symmetric(rho, :U)
    posdef_fix!(ce.posdef, rho)

    return rho
end
function _gerber(ce::CorGerberBasic, X::AbstractMatrix, std_vec::AbstractVector)
    return if ce.normalise
        mean_vec = vec(if isnothing(ce.mean_w)
                           mean(X; dims = 1)
                       else
                           mean(X, ce.mean_w; dims = 1)
                       end)
        _cor_gerber_norm(ce, X, mean_vec, std_vec)
    else
        _cor_gerber(ce, X, std_vec)
    end
end
function _gerber(ce::Union{CorSB, CorGerberSB}, X::AbstractMatrix, std_vec::AbstractVector)
    mean_vec = vec(if isnothing(ce.mean_w)
                       mean(X; dims = 1)
                   else
                       mean(X, ce.mean_w; dims = 1)
                   end)
    return if ce.normalise
        _cor_gerber_norm(ce, X, mean_vec, std_vec)
    else
        _cor_gerber(ce, X, mean_vec, std_vec)
    end
end
function cor_gerber(ce::CorGerber, X::AbstractMatrix)
    std_vec = vec(if isnothing(ce.std_w)
                      std(ce.ve, X; dims = 1)
                  else
                      std(ce.ve, X, ce.std_w; dims = 1)
                  end)
    return Symmetric(_gerber(ce, X, std_vec))
end
function cov_gerber(ce::CorGerber, X::AbstractMatrix)
    std_vec = vec(if isnothing(ce.std_w)
                      std(ce.ve, X; dims = 1)
                  else
                      std(ce.ve, X, ce.std_w; dims = 1)
                  end)
    return Symmetric(_gerber(ce, X, std_vec) .* (std_vec * transpose(std_vec)))
end
function StatsBase.cor(ce::CorGerber, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return cor_gerber(ce, X)
end
function StatsBase.cov(ce::CorGerber, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return cov_gerber(ce, X)
end
#=
function cov_returns(x::AbstractMatrix; iters::Integer = 5, len::Integer = 10,
                     rng = Random.default_rng(), seed::Union{Nothing, <:Integer} = nothing)
    Random.seed!(rng, seed)

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
                 target_ret::Union{Real, <:AbstractVector{<:Real}} = 0.0)
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
                 target_ret::Union{Real, <:AbstractVector{<:Real}} = 0.0)
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
    X = spzeros(rows, cols)
    for j ∈ 1:n
        for i ∈ j:n
            u = spzeros(1, cols)
            col = Int((j - 1) * n + i - (j * (j - 1)) / 2)
            u[col] = 1
            T = spzeros(n, n)
            T[i, j] = 1
            T[j, i] = 1
            X .+= vec(T) * u
        end
    end
    return X
end
function elimination_matrix(n::Int)
    rows = Int(n * (n + 1) / 2)
    cols = n * n
    X = spzeros(rows, cols)
    for j ∈ 1:n
        ej = spzeros(1, n)
        ej[j] = 1
        for i ∈ j:n
            u = spzeros(rows)
            row = Int((j - 1) * n + i - (j * (j - 1)) / 2)
            u[row] = 1
            ei = spzeros(1, n)
            ei[i] = 1
            X .+= kron(u, kron(ej, ei))
        end
    end
    return X
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
=#
abstract type Denoise end
struct NoDenoise <: Denoise end
function denoise!(::NoDenoise, ::PosdefFix, X::AbstractMatrix, q::Real)
    return nothing
end
mutable struct DenoiseFixed{T1, T2, T3, T4} <: Denoise
    detone::Bool
    mkt_comp::T1
    kernel::T2
    m::T3
    n::T4
    args::Tuple
    kwargs::NamedTuple
end
function DenoiseFixed(; detone::Bool = false, mkt_comp::Integer = 1,
                      kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                      n::Integer = 1000, args::Tuple = (), kwargs::NamedTuple = (;))
    return DenoiseFixed{typeof(mkt_comp), typeof(kernel), typeof(m), typeof(n)}(detone,
                                                                                mkt_comp,
                                                                                kernel, m,
                                                                                n, args,
                                                                                kwargs)
end
mutable struct DenoiseSpectral{T1, T2, T3, T4} <: Denoise
    detone::Bool
    mkt_comp::T1
    kernel::T2
    m::T3
    n::T4
    args::Tuple
    kwargs::NamedTuple
end
function DenoiseSpectral(; detone::Bool = false, mkt_comp::Integer = 1,
                         kernel = AverageShiftedHistograms.Kernels.gaussian,
                         m::Integer = 10, n::Integer = 1000, args::Tuple = (),
                         kwargs::NamedTuple = (;))
    return DenoiseSpectral{typeof(mkt_comp), typeof(kernel), typeof(m), typeof(n)}(detone,
                                                                                   mkt_comp,
                                                                                   kernel,
                                                                                   m, n,
                                                                                   args,
                                                                                   kwargs)
end
mutable struct DenoiseShrink{T1, T2, T3, T4, T5} <: Denoise
    detone::Bool
    alpha::T1
    mkt_comp::T2
    kernel::T3
    m::T4
    n::T5
    args::Tuple
    kwargs::NamedTuple
end
function DenoiseShrink(; alpha::Real = 0.0, detone::Bool = false, mkt_comp::Integer = 1,
                       kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                       n::Integer = 1000, args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(zero(alpha) <= alpha <= one(alpha))
    return DenoiseShrink{typeof(alpha), typeof(mkt_comp), typeof(kernel), typeof(m),
                         typeof(n)}(detone, alpha, mkt_comp, kernel, m, n, args, kwargs)
end
function _denoise!(::DenoiseFixed, X::AbstractMatrix, vals::AbstractVector,
                   vecs::AbstractMatrix, num_factors::Integer)
    _vals = copy(vals)
    _vals[1:num_factors] .= sum(_vals[1:num_factors]) / num_factors
    X .= cov2cor(vecs * Diagonal(_vals) * transpose(vecs))
    return nothing
end
function _denoise!(::DenoiseSpectral, X::AbstractMatrix, vals::AbstractVector,
                   vecs::AbstractMatrix, num_factors::Integer)
    _vals = copy(vals)
    _vals[1:num_factors] .= zero(eltype(X))
    X .= cov2cor(vecs * Diagonal(_vals) * transpose(vecs))
    return nothing
end
function _denoise!(ce::DenoiseShrink, X::AbstractMatrix, vals::AbstractVector,
                   vecs::AbstractMatrix, num_factors::Integer)
    # Small
    vals_l = vals[1:num_factors]
    vecs_l = vecs[:, 1:num_factors]

    # Large
    vals_r = vals[(num_factors + 1):end]
    vecs_r = vecs[:, (num_factors + 1):end]

    corr0 = vecs_r * Diagonal(vals_r) * transpose(vecs_r)
    corr1 = vecs_l * Diagonal(vals_l) * transpose(vecs_l)

    X .= corr0 + ce.alpha * corr1 + (one(ce.alpha) - ce.alpha) * Diagonal(corr1)
    return nothing
end
function denoise!(ce::Denoise, posdef::PosdefFix, X::AbstractMatrix, q::Real)
    s = diag(X)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end

    vals, vecs = eigen(X)

    max_val, missing = find_max_eval(vals, q; kernel = ce.kernel, m = ce.m, n = ce.n,
                                     args = ce.args, kwargs = ce.kwargs)

    num_factors = findlast(vals .< max_val)

    _denoise!(ce, X, vals, vecs, num_factors)

    mkt_comp = ce.mkt_comp
    if ce.detone
        @smart_assert(one(size(X, 1)) <= mkt_comp <= size(X, 1))
        mkt_comp -= 1
        _vals = Diagonal(vals)[(end - mkt_comp):end, (end - mkt_comp):end]
        _vecs = vecs[:, (end - mkt_comp):end]
        X .-= _vecs * _vals * transpose(_vecs)
    end

    posdef_fix!(posdef, X)

    if iscov
        StatsBase.cor2cov!(X, s)
    end

    return nothing
end
abstract type MeanEstimator end
abstract type MeanTarget end
struct TargetGM <: MeanTarget end
struct TargetVW <: MeanTarget end
struct TargetSE <: MeanTarget end
@kwdef mutable struct SimpleMean <: MeanEstimator
    w::Union{<:AbstractWeights, Nothing} = nothing
end
function StatsBase.mean(me::SimpleMean, X::AbstractMatrix; dims::Int = 1)
    return vec(isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims))
end
mutable struct MeanJS{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MeanJS(; target::MeanTarget = TargetGM(),
                w::Union{<:AbstractWeights, Nothing} = nothing,
                sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MeanJS{typeof(sigma)}(target, w, sigma)
end
mutable struct MeanBS{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MeanBS(; target::MeanTarget = TargetGM(),
                w::Union{<:AbstractWeights, Nothing} = nothing,
                sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MeanBS{typeof(sigma)}(target, w, sigma)
end
mutable struct MeanBOP{T1} <: MeanEstimator
    target::MeanTarget
    w::Union{<:AbstractWeights, Nothing}
    sigma::T1
end
function MeanBOP(; target::MeanTarget = TargetGM(),
                 w::Union{<:AbstractWeights, Nothing} = nothing,
                 sigma::AbstractMatrix = Matrix{Float64}(undef, 0, 0))
    return MeanBOP{typeof(sigma)}(target, w, sigma)
end
function target_mean(::TargetGM, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma,
                     T::Integer, N::Integer)
    return fill(mean(mu), N)
end
function target_mean(::TargetVW, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma,
                     T::Integer, N::Integer)
    ones = range(one(eltype(sigma)); stop = one(eltype(sigma)), length = N)
    if isnothing(inv_sigma)
        inv_sigma = sigma \ I
    end
    return fill(dot(ones, inv_sigma, mu) / dot(ones, inv_sigma, ones), N)
end
function target_mean(::TargetSE, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma,
                     T::Integer, N::Integer)
    return fill(tr(sigma) / T, N)
end
function StatsBase.mean(me::MeanJS, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    T, N = dims == 1 ? size(X) : size(transpose(X))
    mu = vec(isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims))
    sigma = me.sigma
    b = target_mean(me.target, mu, sigma, nothing, T, N)
    evals = eigvals(sigma)
    alpha = (N * mean(evals) - 2 * maximum(evals)) / dot(mu - b, mu - b) / T
    return (1 - alpha) * mu + alpha * b
end
function StatsBase.mean(me::MeanBS, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    T, N = dims == 1 ? size(X) : size(transpose(X))
    mu = vec(isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims))
    sigma = me.sigma
    inv_sigma = sigma \ I
    b = target_mean(me.target, mu, sigma, inv_sigma, T, N)
    alpha = (N + 2) / ((N + 2) + T * dot(mu - b, inv_sigma, mu - b))
    return (1 - alpha) * mu + alpha * b
end
function StatsBase.mean(me::MeanBOP, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    T, N = dims == 1 ? size(X) : size(transpose(X))
    mu = vec(isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims))
    sigma = me.sigma
    inv_sigma = sigma \ I
    b = target_mean(me.target, mu, sigma, inv_sigma, T, N)
    alpha = (dot(mu, inv_sigma, mu) - N / (T - N)) * dot(b, inv_sigma, b) -
            dot(mu, inv_sigma, b)^2
    alpha /= dot(mu, inv_sigma, mu) * dot(b, inv_sigma, b) - dot(mu, inv_sigma, b)^2
    beta = (1 - alpha) * dot(mu, inv_sigma, b) / dot(mu, inv_sigma, mu)
    return alpha * mu + beta * b
end
@kwdef mutable struct DBHT
    distance::DistanceMethod = DistanceMLP()
    similarity::DBHTSimilarity = DBHTMaxDist()
end
abstract type AbstractJLoGo end
struct NoJLoGo <: AbstractJLoGo end
@kwdef mutable struct JLoGo <: AbstractJLoGo
    DBHT::DBHT = DBHT(;)
end
function jlogo!(::NoJLoGo, ::PosdefFix, ::AbstractMatrix, D = nothing)
    return nothing
end
function jlogo!(je::JLoGo, posdef::PosdefFix, X::AbstractMatrix, D = nothing)
    if isnothing(D)
        s = diag(X)
        iscov = any(.!isone.(s))
        S = if iscov
            s .= sqrt.(s)
            StatsBase.cov2cor(X, s)
        else
            X
        end
        D = dist(je.DBHT.distance, S, nothing)
    end

    S = dbht_similarity(je.DBHT.similarity, S, D)

    separators, cliques = PMFG_T2s(S, 4)[3:4]
    X .= J_LoGo(X, separators, cliques) \ I

    posdef_fix!(posdef, X)

    return nothing
end
abstract type KurtEstimator end
@kwdef mutable struct KurtFull <: KurtEstimator
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise()
    jlogo::AbstractJLoGo = NoJLoGo()
end
@kwdef mutable struct KurtSemi <: KurtEstimator
    target::Union{Real, AbstractVector{<:Real}} = 0.0
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise()
    jlogo::AbstractJLoGo = NoJLoGo()
end
function cokurt(ke::KurtFull, X::AbstractMatrix, mu::AbstractVector)
    T, N = size(X)
    y = X .- transpose(mu)
    o = transpose(range(; start = one(eltype(y)), stop = one(eltype(y)), length = N))
    z = kron(o, y) .* kron(y, o)
    cokurt = transpose(z) * z / T

    posdef_fix!(ke.posdef, cokurt)
    denoise!(ke.denoise, ke.posdef, cokurt, T / N)
    jlogo!(ke.jlogo, ke.posdef, cokurt)

    return cokurt
end
function cokurt(ke::KurtSemi, X::AbstractMatrix, mu::AbstractVector)
    T, N = size(X)
    y = X .- transpose(mu)
    y .= min.(y, ke.target)
    o = transpose(range(; start = one(eltype(y)), stop = one(eltype(y)), length = N))
    z = kron(o, y) .* kron(y, o)
    scokurt = transpose(z) * z / T

    posdef_fix!(ke.posdef, scokurt)
    denoise!(ke.denoise, ke.posdef, scokurt, T / N)
    jlogo!(ke.jlogo, ke.posdef, scokurt)

    return scokurt
end
abstract type SkewEstimator end
struct SkewFull <: SkewEstimator
    # posdef::PosdefFix = PosdefNearest(;)
    # denoise::Denoise = NoDenoise()
    # jlogo::AbstractJLoGo = NoJLoGo()
end
@kwdef mutable struct SkewSemi <: SkewEstimator
    target::Union{Real, AbstractVector{<:Real}} = 0.0
    # posdef::PosdefFix = PosdefNearest(;)
    # denoise::Denoise = NoDenoise()
    # jlogo::AbstractJLoGo = NoJLoGo()
end
function coskew(se::SkewFull, X::AbstractMatrix, mu::AbstractVector)
    T, N = size(X)
    y = X .- transpose(mu)
    o = transpose(range(; start = one(eltype(y)), stop = one(eltype(y)), length = N))
    z = kron(o, y) .* kron(y, o)
    coskew = transpose(X) * z / T

    V = zeros(eltype(y), N, N)
    for i ∈ 1:N
        j = (i - 1) * N + 1
        k = i * N
        coskew_jk = coskew[:, j:k]
        vals, vecs = eigen(coskew_jk)
        vals = clamp.(real.(vals), -Inf, 0) .+ clamp.(imag.(vals), -Inf, 0)im
        V .-= real(vecs * Diagonal(vals) * transpose(vecs))
        # denoise!(se.denoise, se.posdef, -real(vecs * Diagonal(vals) * transpose(vecs)),
        #          T / N)
        # jlogo!(se.jlogo, se.posdef, -real(vecs * Diagonal(vals) * transpose(vecs)))
    end

    return coskew, V
end
function coskew(se::SkewSemi, X::AbstractMatrix, mu::AbstractVector)
    T, N = size(X)
    y = X .- transpose(mu)
    y .= min.(y, se.target)
    o = transpose(range(; start = one(eltype(y)), stop = one(eltype(y)), length = N))
    z = kron(o, y) .* kron(y, o)
    scoskew = transpose(X) * z / T

    SV = zeros(eltype(y), N, N)
    for i ∈ 1:N
        j = (i - 1) * N + 1
        k = i * N
        scoskew_jk = scoskew[:, j:k]
        vals, vecs = eigen(scoskew_jk)
        vals = clamp.(real.(vals), -Inf, 0) .+ clamp.(imag.(vals), -Inf, 0)im
        SV .-= real(vecs * Diagonal(vals) * transpose(vecs))
        # denoise!(se.denoise, NoPosdef(), -real(vecs * Diagonal(vals) * transpose(vecs)),
        #          T / N)
        # jlogo!(se.jlogo, NoPosdef(), -real(vecs * Diagonal(vals) * transpose(vecs)))
    end

    return scoskew, SV
end

@kwdef mutable struct CovType
    ce::PortfolioOptimiserCovCor = CovFull(;)
    posdef::PosdefFix = PosdefNearest(;)
    denoise::Denoise = NoDenoise()
    jlogo::AbstractJLoGo = NoJLoGo()
end
function StatsBase.cov(ce::CovType, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = Matrix(cov(ce.ce, X))
    posdef_fix!(ce.posdef, sigma)
    denoise!(ce.denoise, ce.posdef, sigma, size(X, 1) / size(X, 2))
    jlogo!(ce.jlogo, ce.posdef, sigma)

    return Symmetric(sigma)
end
function StatsBase.cor(ce::CovType, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = Matrix(cor(ce.ce, X))
    posdef_fix!(ce.posdef, rho)
    denoise!(ce.denoise, ce.posdef, rho, size(X, 1) / size(X, 2))
    jlogo!(ce.jlogo, ce.posdef, rho)

    return Symmetric(rho)
end

function asset_statistics2!(portfolio::AbstractPortfolio; cov_type::CovType = CovType(;),
                            set_cov::Bool = true, mu_type::MeanEstimator = SimpleMean(;),
                            set_mu::Bool = true, kurt_type::KurtFull = KurtFull(;),
                            set_kurt::Bool = true, skurt_type::KurtSemi = KurtSemi(;),
                            set_skurt::Bool = true, skew_type::SkewFull = SkewFull(;),
                            set_skew::Bool = true, sskew_type::SkewSemi = SkewSemi(;),
                            set_sskew::Bool = true, cor_type::CovType = CovType(;),
                            set_cor::Bool = true,
                            dist_type::DistanceMethod = DistanceDefault(),
                            set_dist::Bool = true)
    returns = portfolio.returns

    if set_cov || set_mu && hasproperty(mu_type, :sigma)
        sigma = cov(cov_type, returns)
        if set_cov
            portfolio.cov = sigma
        end
    end
    if set_mu || set_kurt || set_skurt || set_skew || set_sskew
        if hasproperty(mu_type, :sigma)
            mu_type.sigma = sigma
        end
        mu = mean(mu_type, returns)
        if set_mu
            portfolio.mu = mu
        end
    end
    if set_kurt || set_skurt
        if set_kurt
            portfolio.kurt = cokurt(kurt_type, returns, mu)
        end
        if set_skurt
            portfolio.skurt = cokurt(skurt_type, returns, mu)
        end
        portfolio.L_2, portfolio.S_2 = dup_elim_sum_matrices(size(returns, 2))[2:3]
    end
    if set_skew
        portfolio.skew, portfolio.V = coskew(skew_type, returns, mu)
    end
    if set_sskew
        portfolio.sskew, portfolio.SV = coskew(sskew_type, returns, mu)
    end

    if isa(portfolio, HCPortfolio)
        if set_cor || set_dist
            rho = cor(cor_type, returns)
            if set_cor
                portfolio.cor = rho
            end
        end

        if set_dist
            if isa(dist_type, DistanceDefault)
                dist_type = if isa(cor_type.ce, CorMutualInfo)
                    DistanceVarInfo(; bins = cor_type.ce.bins,
                                    normalise = cor_type.ce.normalise)
                elseif isa(cov_type.ce, CorLTD)
                    DistanceLog()
                else
                    DistanceMLP()
                end
            end

            if hasproperty(cor_type.ce, :absolute) && hasproperty(dist_type, :absolute)
                dist_type.absolute = cor_type.ce.absolute
            end
            portfolio.dist = dist(dist_type, rho, returns)
        end
    end

    return nothing
end

export CovFull, CovSemi, CorSpearman, CorKendall, CorMutualInfo, CorDistance, CorLTD,
       CorGerber0, CorGerber1, CorGerber2, CorSB0, CorSB1, CorGerberSB0, CorGerberSB1,
       DistanceMLP, dist, CovType, DistanceVarInfo, BinKnuth, BinFreedman, BinScott, BinHGR,
       DistanceLog, DistanceMLP2, MeanEstimator, MeanTarget, TargetGM, TargetVW, TargetSE,
       SimpleMean, MeanJS, MeanBS, MeanBOP, SimpleVariance, asset_statistics2!, JLoGo,
       SkewFull, SkewSemi, KurtFull, KurtSemi, DenoiseFixed, DenoiseSpectral, DenoiseShrink
