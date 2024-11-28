function StatsBase.std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing)
    return std(X; dims = dims, corrected = ve.corrected, mean = mean)
end
function StatsBase.std(ve::SimpleVariance, X::AbstractArray, w::AbstractWeights;
                       dims::Int = 1, mean = nothing)
    return std(X, w, dims; corrected = ve.corrected, mean = mean)
end
function StatsBase.var(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing)
    return var(X; dims = dims, corrected = ve.corrected, mean = mean)
end
function StatsBase.var(ve::SimpleVariance, X::AbstractArray, w::AbstractWeights;
                       dims::Int = 1, mean = nothing)
    return var(X, w, dims; corrected = ve.corrected, mean = mean)
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
function StatsBase.cov(ce::CovSemi, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    target = ce.target
    mu = ce.mu
    if isnothing(mu)
        mu = vec(if isnothing(ce.mu_w)
                     mean(X; dims = 1)
                 else
                     mean(X, ce.mu_w; dims = 1)
                 end)
    end
    X = min.(X .- transpose(mu), target)

    return Symmetric(if isnothing(ce.cov_w)
                         cov(ce.ce, X; mean = zero(eltype(X)))
                     else
                         cov(ce.ce, X, ce.cov_w; mean = zero(eltype(X)))
                     end)
end
function StatsBase.cor(ce::CovSemi, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    target = ce.target
    mu = ce.mu
    if isnothing(mu)
        mu = vec(if isnothing(ce.mu_w)
                     mean(X; dims = 1)
                 else
                     mean(X, ce.mu_w; dims = 1)
                 end)
    end
    X = min.(X .- transpose(mu), target)

    rho = Symmetric(try
                        if isnothing(ce.cov_w)
                            cor(ce.ce, X; mean = zero(eltype(X)))
                        else
                            cor(ce.ce, X, ce.cov_w; mean = zero(eltype(X)))
                        end
                    catch
                        StatsBase.cov2cor(Matrix(if isnothing(ce.cov_w)
                                                     cov(ce.ce, X; mean = zero(eltype(X)))
                                                 else
                                                     cov(ce.ce, X, ce.cov_w;
                                                         mean = zero(eltype(X)))
                                                 end))
                    end)
    return !ce.absolute ? rho : abs.(rho)
end
function StatsBase.cor(ce::CorSpearman, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = corspearman(X)
    return Symmetric(cov2cor(Matrix(!ce.absolute ? rho : abs.(rho))))
end
function StatsBase.cor(ce::CorKendall, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = corkendall(X)
    return Symmetric(cov2cor(Matrix(!ce.absolute ? rho : abs.(rho))))
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
function mutual_info(X::AbstractMatrix, bins::Union{<:AbstractBins, <:Integer} = HGR(),
                     normalise::Bool = true)
    T, N = size(X)
    mut_mtx = Matrix{eltype(X)}(undef, N, N)

    bin_width_func = _bin_width_func(bins)

    for j ∈ axes(X, 2)
        xj = X[:, j]
        for i ∈ 1:j
            xi = X[:, i]
            nbins = calc_num_bins(bins, xj, xi, j, i, bin_width_func, T)
            ex, ey, hxy = calc_hist_data(xj, xi, nbins)

            mut_ixy = _mutual_info(hxy)
            if normalise
                mut_ixy /= min(ex, ey)
            end

            mut_ixy = clamp(mut_ixy, zero(eltype(X)), Inf)

            mut_mtx[i, j] = mut_ixy
        end
    end

    return Symmetric(mut_mtx, :U)
end
function StatsBase.cor(ce::CorMutualInfo, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return mutual_info(X, ce.bins, ce.normalise)
end
function StatsBase.cov(ce::CorMutualInfo, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = vec(if isnothing(ce.w)
                      std(ce.ve, X; dims = 1)
                  else
                      std(ce.ve, X, ce.w; dims = 1)
                  end)
    return Symmetric(mutual_info(X, ce.bins, ce.normalise) .*
                     (std_vec * transpose(std_vec)))
end
function cor_distance(ce::CovDistance, v1::AbstractVector, v2::AbstractVector)
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

    return sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))
end
function cor_distance(ce::CovDistance, X::AbstractMatrix)
    N = size(X, 2)

    rho = Matrix{eltype(X)}(undef, N, N)
    for j ∈ axes(X, 2)
        xj = X[:, j]
        for i ∈ 1:j
            rho[i, j] = cor_distance(ce, X[:, i], xj)
        end
    end

    return Symmetric(rho, :U)
end
function StatsBase.cor(ce::CovDistance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return cor_distance(ce::CovDistance, X::AbstractMatrix)
end
function cov_distance(ce::CovDistance, v1::AbstractVector, v2::AbstractVector)
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

    return sqrt(dcov2_xy)
end
function cov_distance(ce::CovDistance, X::AbstractMatrix)
    N = size(X, 2)

    rho = Matrix{eltype(X)}(undef, N, N)
    for j ∈ axes(X, 2)
        xj = X[:, j]
        for i ∈ 1:j
            rho[i, j] = cov_distance(ce, X[:, i], xj)
        end
    end

    return Symmetric(rho, :U)
end
function StatsBase.cov(ce::CovDistance, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return Symmetric(cov_distance(ce, X))
end
function lower_tail_dependence(X::AbstractMatrix, alpha::Real = 0.05)
    T, N = size(X)
    k = ceil(Int, T * alpha)
    rho = Matrix{eltype(X)}(undef, N, N)

    if k > 0
        for j ∈ axes(X, 2)
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
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return lower_tail_dependence(X, ce.alpha)
end
function StatsBase.cov(ce::CorLTD, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = vec(if isnothing(ce.w)
                      std(ce.ve, X; dims = 1)
                  else
                      std(ce.ve, X, ce.w; dims = 1)
                  end)
    return lower_tail_dependence(X, ce.alpha) .* (std_vec * transpose(std_vec))
end
function _cor_gerber_norm(ce::CorGerber0, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold

    for j ∈ axes(X, 2)
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

    for j ∈ axes(X, 2)
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

    for j ∈ axes(X, 2)
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

    for j ∈ axes(X, 2)
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

    for i ∈ axes(X, 2)
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

    for i ∈ axes(X, 2)
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
function _cor_gerber_norm(ce::CorSB0, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n

    for j ∈ axes(X, 2)
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

    for j ∈ axes(X, 2)
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

    for j ∈ axes(X, 2)
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

    for j ∈ axes(X, 2)
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

    for j ∈ axes(X, 2)
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

    for j ∈ axes(X, 2)
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

    for j ∈ axes(X, 2)
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

    for j ∈ axes(X, 2)
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
function block_vec_pq(A, p, q)
    mp, nq = size(A)

    if !(mod(mp, p) == 0 && mod(nq, q) == 0)
        throw(DimensionMismatch("size(A) = $(size(A)), must be integer multiples of (p, q) = ($p, $q)"))
    end

    m = Int(mp / p)
    n = Int(nq / q)

    A_vec = Matrix{eltype(A)}(undef, m * n, p * q)
    for j ∈ 0:(n - 1)
        Aj = Matrix{eltype(A)}(undef, m, p * q)
        for i ∈ 0:(m - 1)
            Aij = vec(A[(1 + (i * p)):((i + 1) * p), (1 + (j * q)):((j + 1) * q)])
            Aj[i + 1, :] .= Aij
        end
        A_vec[(1 + (j * m)):((j + 1) * m), :] .= Aj
    end

    return A_vec
end
function duplication_matrix(n::Int)
    m   = Int(n * (n + 1) / 2)
    nsq = n^2
    v   = zeros(Int, nsq)
    r   = 1
    a   = 1
    for i ∈ 1:n
        b = i
        for j ∈ 0:(i - 2)
            v[r] = b
            b    += n - j - 1
            r    += 1
        end

        for j ∈ 0:(n - i)
            v[r] = a + j
            r    += 1
        end
        a += n - i + 1
    end

    return sparse(1:nsq, v, 1, nsq, m)
end
function elimination_matrix(n::Int)
    m   = Int(n * (n + 1) / 2)
    nsq = n^2
    v   = zeros(Int, m)
    r   = 1
    a   = 1
    b   = 0
    for i ∈ 1:n
        for j ∈ 0:(n - i)
            v[r] = a + j + b
            r += 1
        end
        a += n - i + 1
        b += i
    end

    return sparse(1:m, v, 1, m, nsq)
end
function summation_matrix(n::Int)
    m = Int(n * (n + 1) / 2)
    nsq = n^2
    r = 0
    a = 1
    v1 = zeros(Int, nsq)
    v2 = zeros(Int, nsq)
    rows2 = zeros(Int, nsq)
    cm = 0

    for i ∈ 1:n
        r += i - 1
        for j ∈ 0:(n - i)
            v1[r + j + 1] = a + j + cm
        end
        for j ∈ 1:(n - i)
            v2[r + j + 1] = a + j + cm
            rows2[r + j + 1] = a + j
        end
        r += n - i + 1
        a += n - i + 1
        cm += i
    end
    v1 = v1[.!iszero.(v1)]
    v2 = v2[.!iszero.(v2)]
    rows2 = rows2[.!iszero.(rows2)]

    a = sparse(1:m, v1, 1, m, nsq)
    b = sparse(rows2, v2, 1, m, nsq)
    return a + b
end
function dup_elim_sum_matrices(n::Int)
    m   = Int(n * (n + 1) / 2)
    nsq = n^2
    v1  = zeros(Int, nsq)
    v2  = zeros(Int, m)
    r1  = 1
    r2  = 1
    a   = 1
    b2  = 0
    for i ∈ 1:n
        b1 = i
        for j ∈ 0:(i - 2)
            v1[r1] = b1
            b1 += n - j - 1
            r1 += 1
        end

        for j ∈ 0:(n - i)
            v1[r1] = a + j
            v2[r2] = a + j + b2
            r1 += 1
            r2 += 1
        end
        a += n - i + 1
        b2 += i
    end

    d = sparse(1:nsq, v1, 1, nsq, m)
    l = sparse(1:m, v2, 1, m, nsq)
    s = transpose(d) * d * l

    return d, l, s
end
"""
```
logo!(::NoLoGo, ::PosdefFix, ::AbstractMatrix, D = nothing)
```
"""
function logo!(::NoLoGo, ::PosdefFix, ::AbstractMatrix, D = nothing)
    return nothing
end
"""
```
logo!(je::LoGo, posdef::PosdefFix, X::AbstractMatrix, D = nothing)
```
"""
function logo!(je::LoGo, posdef::PosdefFix, X::AbstractMatrix, D = nothing)
    if isnothing(D)
        s = diag(X)
        iscov = any(.!isone.(s))
        S = if iscov
            s .= sqrt.(s)
            StatsBase.cov2cor(X, s)
        else
            X
        end
        D = dist(je.distance, S, nothing)
    end

    S = dbht_similarity(je.similarity, S, D)
    separators, cliques = PMFG_T2s(S, 4)[3:4]
    X .= J_LoGo(X, separators, cliques) \ I

    posdef_fix!(posdef, X)

    return nothing
end
function cokurt(ke::KurtFull, X::AbstractMatrix, mu::AbstractVector)
    T, N = size(X)
    y = X .- transpose(mu)
    o = transpose(range(; start = one(eltype(y)), stop = one(eltype(y)), length = N))
    z = kron(o, y) .* kron(y, o)
    cokurt = transpose(z) * z / T

    posdef_fix!(ke.posdef, cokurt)
    denoise!(ke.denoise, ke.posdef, cokurt, T / N)
    logo!(ke.logo, ke.posdef, cokurt)

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
    logo!(ke.logo, ke.posdef, scokurt)

    return scokurt
end
"""
```
coskew(::SkewFull, X::AbstractMatrix, mu::AbstractVector)
```
"""
function coskew(::SkewFull, X::AbstractMatrix, mu::AbstractVector)
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
        # logo!(se.logo, se.posdef, -real(vecs * Diagonal(vals) * transpose(vecs)))
    end

    return coskew, V
end
"""
```
coskew(se::SkewSemi, X::AbstractMatrix, mu::AbstractVector)
```
"""
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
        # logo!(se.logo, NoPosdef(), -real(vecs * Diagonal(vals) * transpose(vecs)))
    end

    return scoskew, SV
end
function StatsBase.cov(ce::PortCovCor, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = Matrix(cov(ce.ce, X))
    posdef_fix!(ce.posdef, sigma)
    denoise!(ce.denoise, ce.posdef, sigma, size(X, 1) / size(X, 2))
    logo!(ce.logo, ce.posdef, sigma)

    return Symmetric(sigma)
end
function StatsBase.cor(ce::PortCovCor, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = try
        Matrix(cor(ce.ce, X))
    catch
        cov2cor(Matrix(cov(ce.ce, X)))
    end
    posdef_fix!(ce.posdef, rho)
    denoise!(ce.denoise, ce.posdef, rho, size(X, 1) / size(X, 2))
    logo!(ce.logo, ce.posdef, rho)

    return Symmetric(rho)
end

export mutual_info, cor_distance, cov_distance, lower_tail_dependence, cov_gerber,
       cor_gerber, duplication_matrix, elimination_matrix, summation_matrix,
       dup_elim_sum_matrices, logo!, cokurt, coskew
