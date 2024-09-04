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

"""
```
_posdef_fix!(method::PosdefNearest, X::AbstractMatrix)
```

Overload this for other posdef fix methods.
"""
function _posdef_fix!(method::PosdefNearest, X::AbstractMatrix)
    NearestCorrelationMatrix.nearest_cor!(X, method)
    return nothing
end
function posdef_fix!(::NoPosdef, ::AbstractMatrix)
    return nothing
end
"""
```
posdef_fix!(method::PosdefFix, X::AbstractMatrix)
```
"""
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
"""
```
dbht_similarity(::DBHTExp, S, D)
```
"""
function dbht_similarity(::DBHTExp, S, D)
    return exp.(-D)
end
function dbht_similarity(::DBHTMaxDist, S, D)
    return ceil(maximum(D)^2) .- D .^ 2
end
function _dist(de::DistanceMLP, X::AbstractMatrix, ::Any)
    return Symmetric(sqrt.(if !de.absolute
                               clamp!((one(eltype(X)) .- X) / 2, zero(eltype(X)),
                                      one(eltype(X)))
                           else
                               clamp!(one(eltype(X)) .- X, zero(eltype(X)), one(eltype(X)))
                           end))
end
function _dist(de::DistanceSqMLP, X::AbstractMatrix, ::Any)
    _X = sqrt.(if !de.absolute
                   clamp!((one(eltype(X)) .- X) / 2, zero(eltype(X)), one(eltype(X)))
               else
                   clamp!(one(eltype(X)) .- X, zero(eltype(X)), one(eltype(X)))
               end)

    return Distances.pairwise(de.distance, _X, de.args...; de.kwargs...)
end
function _dist(::DistanceLog, X::AbstractMatrix, ::Any)
    return -log.(X)
end
"""
```
dist(de::DistanceMethod, X, Y)
```
"""
function dist(de::DistanceMethod, X, Y)
    return _dist(de, X, Y)
end
function StatsBase.std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return std(X; dims = dims, corrected = ve.corrected, mean = mean)
end
function StatsBase.std(ve::SimpleVariance, X::AbstractMatrix, w::AbstractWeights;
                       dims::Int = 1, mean = nothing)
    return std(X, w, dims; corrected = ve.corrected, mean = mean)
end
function StatsBase.var(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return var(X; dims = dims, corrected = ve.corrected, mean = mean)
end
function StatsBase.var(ve::SimpleVariance, X::AbstractMatrix, w::AbstractWeights;
                       dims::Int = 1, mean = nothing)
    return var(X, w, dims; corrected = ve.corrected, mean = mean)
end
# # Correlation Matrices
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
    @smart_assert(dims ∈ (1, 2))
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
function Base.setproperty!(obj::DistanceVarInfo, sym::Symbol, val)
    if sym == :bins
        if isa(val, Integer)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end
function Base.setproperty!(obj::CorMutualInfo, sym::Symbol, val)
    if sym == :bins
        if isa(val, Integer)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end
function _bin_width_func(::Knuth)
    return pyimport("astropy.stats").knuth_bin_width
end
function _bin_width_func(::Freedman)
    return pyimport("astropy.stats").freedman_bin_width
end
function _bin_width_func(::Scott)
    return pyimport("astropy.stats").scott_bin_width
end
function _bin_width_func(::Union{HGR, <:Integer})
    return nothing
end
function calc_num_bins(::AstroBins, xj::AbstractVector, xi::AbstractVector, j::Integer,
                       i::Integer, bin_width_func, ::Any)
    k1 = (maximum(xj) - minimum(xj)) / bin_width_func(xj)
    return round(Int, if j != i
                     k2 = (maximum(xi) - minimum(xi)) / bin_width_func(xi)
                     max(k1, k2)
                 else
                     k1
                 end)
end
function calc_num_bins(::HGR, xj::AbstractVector, xi::AbstractVector, j::Integer,
                       i::Integer, ::Any, T::Integer)
    corr = cor(xj, xi)
    return round(Int, if isone(corr)
                     z = cbrt(8 + 324 * T + 12 * sqrt(36 * T + 729 * T^2))
                     z / 6 + 2 / (3 * z) + 1 / 3
                 else
                     sqrt(1 + sqrt(1 + 24 * T / (1 - corr^2))) / sqrt(2)
                 end)
end
function calc_num_bins(bins::Integer, args...)
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
#=
function mutual_variation_info(X::AbstractMatrix,
                               bins::Union{<:AbstractBins, <:Integer} = Knuth(),
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

            # if abs(mut_ixy) < eps(typeof(mut_ixy)) || mut_ixy < zero(eltype(X))
            #     mut_ixy = zero(eltype(X))
            # end
            # if abs(var_ixy) < eps(typeof(var_ixy)) || var_ixy < zero(eltype(X))
            #     var_ixy = zero(eltype(X))
            # end

            mut_ixy = clamp(mut_ixy, zero(eltype(X)), Inf)
            var_ixy = clamp(var_ixy, zero(eltype(X)), Inf)

            mut_mtx[i, j] = mut_ixy
            var_mtx[i, j] = var_ixy
        end
    end

    return Symmetric(mut_mtx, :U), Symmetric(var_mtx, :U)
end
=#
function mutual_info(X::AbstractMatrix, bins::Union{<:AbstractBins, <:Integer} = HGR(),
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
                mut_ixy /= min(ex, ey)
            end

            mut_ixy = clamp(mut_ixy, zero(eltype(X)), Inf)

            mut_mtx[i, j] = mut_ixy
        end
    end

    return Symmetric(mut_mtx, :U)
end
function variation_info(X::AbstractMatrix, bins::Union{<:AbstractBins, <:Integer} = HGR(),
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

            var_ixy = clamp(var_ixy, zero(eltype(X)), Inf)

            var_mtx[i, j] = var_ixy
        end
    end

    return Symmetric(var_mtx, :U)
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
function _dist(ce::DistanceVarInfo, ::Any, Y::AbstractMatrix)
    return variation_info(Y, ce.bins, ce.normalise)
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

    return sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))
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
    @smart_assert(dims ∈ (1, 2))
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

    return sqrt(dcov2_xy)
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
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return Symmetric(cov_distance(ce, X))
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
function duplication_matrix(n::Int)
    # cols = Int(n * (n + 1) / 2)
    # rows = n * n
    # X = spzeros(rows, cols)
    # for j ∈ 1:n
    #     for i ∈ j:n
    #         u = spzeros(1, cols)
    #         col = Int((j - 1) * n + i - (j * (j - 1)) / 2)
    #         u[col] = 1
    #         T = spzeros(n, n)
    #         T[i, j] = 1
    #         T[j, i] = 1
    #         X .+= vec(T) * u
    #     end
    # end
    # return X

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
    # rows = Int(n * (n + 1) / 2)
    # cols = n * n
    # X = spzeros(rows, cols)
    # for j ∈ 1:n
    #     ej = spzeros(1, n)
    #     ej[j] = 1
    #     for i ∈ j:n
    #         u = spzeros(rows)
    #         row = Int((j - 1) * n + i - (j * (j - 1)) / 2)
    #         u[row] = 1
    #         ei = spzeros(1, n)
    #         ei[i] = 1
    #         X .+= kron(u, kron(ej, ei))
    #     end
    # end
    # return X

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
    # d = duplication_matrix(n)
    # l = elimination_matrix(n)
    # s = transpose(d) * d * l
    # return s

    m   = Int(n * (n + 1) / 2)
    nsq = n^2
    v   = zeros(Int, nsq)
    v2  = zeros(Int, m)
    r1  = 1
    r2  = 1
    a   = 1
    b2  = 0
    for i ∈ 1:n
        b1 = i
        for j ∈ 0:(i - 2)
            v[r1] = b1
            b1    += n - j - 1
            r1    += 1
        end

        for j ∈ 0:(n - i)
            v[r1] = a + j
            v2[r2] = a + j + b2
            r1 += 1
            r2 += 1
        end
        a += n - i + 1
        b2 += i
    end

    d = sparse(1:nsq, v, 1, nsq, m)
    l = sparse(1:m, v2, 1, m, nsq)
    s = transpose(d) * d * l

    return s
end
function dup_elim_sum_matrices(n::Int)
    # d = duplication_matrix(n)
    # l = elimination_matrix(n)
    # s = transpose(d) * d * l

    m   = Int(n * (n + 1) / 2)
    nsq = n^2
    v   = zeros(Int, nsq)
    v2  = zeros(Int, m)
    r1  = 1
    r2  = 1
    a   = 1
    b2  = 0
    for i ∈ 1:n
        b1 = i
        for j ∈ 0:(i - 2)
            v[r1] = b1
            b1    += n - j - 1
            r1    += 1
        end

        for j ∈ 0:(n - i)
            v[r1] = a + j
            v2[r2] = a + j + b2
            r1 += 1
            r2 += 1
        end
        a += n - i + 1
        b2 += i
    end

    d = sparse(1:nsq, v, 1, nsq, m)
    l = sparse(1:m, v2, 1, m, nsq)
    s = transpose(d) * d * l

    return d, l, s
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
function errPDF(x, vals; kernel = AverageShiftedHistograms.Kernels.gaussian, m = 10,
                n = 1000, q = 1000)
    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    rg = range(e_min, e_max; length = n)
    pdf1 = q ./ (2 * pi * x * rg) .* sqrt.(clamp.((e_max .- rg) .* (rg .- e_min), 0, Inf))
    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    res = ash(vals; rng = range(e_min, e_max; length = n), kernel = kernel, m = m)
    pdf2 = [AverageShiftedHistograms.pdf(res, i) for i ∈ pdf1]
    pdf2[.!isfinite.(pdf2)] .= 0.0
    sse = sum((pdf2 - pdf1) .^ 2)
    return sse
end
function find_max_eval(vals, q; kernel = AverageShiftedHistograms.Kernels.gaussian,
                       m::Integer = 10, n::Integer = 1000, args = (), kwargs = (;))
    res = Optim.optimize(x -> errPDF(x, vals; kernel = kernel, m = m, n = n, q = q), 0.0,
                         1.0, args...; kwargs...)

    x = Optim.converged(res) ? Optim.minimizer(res) : 1.0

    e_max = x * (1.0 + sqrt(1.0 / q))^2

    return e_max, x
end
function denoise!(::NoDenoise, ::PosdefFix, X::AbstractMatrix, q::Real)
    return nothing
end
"""
```
denoise!(ce::Denoise, posdef::PosdefFix, X::AbstractMatrix, q::Real)
```
"""
function denoise!(ce::Denoise, posdef::PosdefFix, X::AbstractMatrix, q::Real)
    s = diag(X)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end

    vals, vecs = eigen(X)

    max_val = find_max_eval(vals, q; kernel = ce.kernel, m = ce.m, n = ce.n, args = ce.args,
                            kwargs = ce.kwargs)[1]

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
function StatsBase.mean(me::MuSimple, X::AbstractMatrix; dims::Int = 1)
    return isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims)
end
"""
```
target_mean(::GM, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma, T::Integer,
                     N::Integer)
```
"""
function target_mean(::GM, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma, T::Integer,
                     N::Integer)
    return fill(mean(mu), N)
end
function target_mean(::VW, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma, T::Integer,
                     N::Integer)
    ones = range(one(eltype(sigma)); stop = one(eltype(sigma)), length = N)
    if isnothing(inv_sigma)
        inv_sigma = sigma \ I
    end
    return fill(dot(ones, inv_sigma, mu) / dot(ones, inv_sigma, ones), N)
end
function target_mean(::SE, mu::AbstractVector, sigma::AbstractMatrix, inv_sigma, T::Integer,
                     N::Integer)
    return fill(tr(sigma) / T, N)
end
function StatsBase.mean(me::MuJS, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    T, N = dims == 1 ? size(X) : size(transpose(X))
    mu = vec(isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims))
    sigma = me.sigma
    b = target_mean(me.target, mu, sigma, nothing, T, N)
    evals = eigvals(sigma)
    alpha = (N * mean(evals) - 2 * maximum(evals)) / dot(mu - b, mu - b) / T
    return (1 - alpha) * mu + alpha * b
end
function StatsBase.mean(me::MuBS, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    T, N = dims == 1 ? size(X) : size(transpose(X))
    mu = vec(isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims))
    sigma = me.sigma
    inv_sigma = sigma \ I
    b = target_mean(me.target, mu, sigma, inv_sigma, T, N)
    alpha = (N + 2) / ((N + 2) + T * dot(mu - b, inv_sigma, mu - b))
    return (1 - alpha) * mu + alpha * b
end
function StatsBase.mean(me::MuBOP, X::AbstractMatrix; dims::Int = 1)
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
function logo!(::NoLoGo, ::PosdefFix, ::AbstractMatrix, D = nothing)
    return nothing
end
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
    rho = Matrix(cor(ce.ce, X))
    posdef_fix!(ce.posdef, rho)
    denoise!(ce.denoise, ce.posdef, rho, size(X, 1) / size(X, 2))
    logo!(ce.logo, ce.posdef, rho)

    return Symmetric(rho)
end
function _get_default_dist(dist_type::DistanceMethod, cor_type::PortfolioOptimiserCovCor)
    if isa(dist_type, DistanceCanonical)
        dist_type = if isa(cor_type.ce, CorMutualInfo)
            DistanceVarInfo(; bins = cor_type.ce.bins, normalise = cor_type.ce.normalise)
        elseif isa(cor_type.ce, CorLTD)
            DistanceLog()
        else
            DistanceMLP()
        end
    end

    if hasproperty(cor_type.ce, :absolute) && hasproperty(dist_type, :absolute)
        dist_type.absolute = cor_type.ce.absolute
    end

    return dist_type
end
function _bootstrap_func(::StationaryBS)
    return pyimport("arch.bootstrap").StationaryBS
end
function _bootstrap_func(::CircularBS)
    return pyimport("arch.bootstrap").CircularBlockBootstrap
end
function _bootstrap_func(::MovingBS)
    return pyimport("arch.bootstrap").MovingBlockBootstrap
end
function _sigma_mu(X::AbstractArray, cov_type::PortfolioOptimiserCovCor,
                   mu_type::MeanEstimator)
    sigma = Matrix(cov(cov_type, X))
    if hasproperty(mu_type, :sigma)
        mu_type.sigma = sigma
    end
    mu = vec(mean(mu_type, X))

    return sigma, mu
end
function gen_bootstrap(method::ArchWC, cov_type::PortfolioOptimiserCovCor,
                       mu_type::MeanEstimator, X::AbstractMatrix)
    covs = Vector{Matrix{eltype(X)}}(undef, 0)
    sizehint!(covs, method.n_sim)
    mus = Vector{Vector{eltype(X)}}(undef, 0)
    sizehint!(mus, method.n_sim)

    bootstrap_func = _bootstrap_func(method.bootstrap)
    gen = bootstrap_func(method.block_size, X; seed = method.seed)
    for data ∈ gen.bootstrap(method.n_sim)
        A = data[1][1]
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
function Base.setproperty!(obj::PVal, sym::Symbol, val)
    if sym == :threshold
        @smart_assert(zero(val) < val < one(val))
    end
    return setfield!(obj, sym, val)
end
function MultivariateStats.fit(method::PCATarget, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PCA, X; method.kwargs...)
end
function MultivariateStats.fit(method::PPCATarget, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PPCA, X; method.kwargs...)
end
function prep_dim_red_reg(method::PCAReg, x::DataFrame)
    N = nrow(x)
    X = transpose(Matrix(x))

    X_std = StatsBase.standardize(StatsBase.ZScoreTransform, X; dims = 2)

    model = fit(method.target, X_std)
    Xp = transpose(predict(model, X_std))
    Vp = projection(model)
    x1 = [ones(N) Xp]

    return X, x1, Vp
end
function _regression(method::PCAReg, X::AbstractMatrix, x1::AbstractMatrix,
                     Vp::AbstractMatrix, y::AbstractVector)
    avg = vec(if isnothing(method.mean_w)
                  mean(X; dims = 2)
              else
                  mean(X, method.mean_w; dims = 2)
              end)
    sdev = vec(if isnothing(method.std_w)
                   std(method.ve, X; dims = 2)
               else
                   std(method.ve, X, method.std_w; dims = 2)
               end)

    fit_result = lm(x1, y)
    beta_pc = coef(fit_result)[2:end]

    beta = Vp * beta_pc ./ sdev
    beta0 = mean(y) - dot(beta, avg)
    pushfirst!(beta, beta0)

    return beta
end
function regression(method::PCAReg, x::DataFrame, y::DataFrame)
    features = names(x)
    rows = ncol(y)
    cols = ncol(x) + 1

    loadings = zeros(rows, cols)

    X, x1, Vp = prep_dim_red_reg(method, x)
    for i ∈ 1:rows
        beta = _regression(method, X, x1, Vp, y[!, i])
        loadings[i, :] .= beta
    end

    return hcat(DataFrame(; tickers = names(y)), DataFrame(loadings, ["const"; features]))
end
function _regression(::FReg, criterion::PVal, x::DataFrame, y::AbstractVector)
    ovec = ones(length(y))
    namesx = names(x)

    threshold = criterion.threshold

    included = String[]
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

    return included
end
function _regression(::BReg, criterion::PVal, x::DataFrame, y::AbstractVector)
    ovec = ones(length(y))
    fit_result = lm([ovec Matrix(x)], y)

    included = names(x)
    namesx = names(x)

    threshold = criterion.threshold

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

        push!(included, new_feature)
    end

    return included
end
function _regression_criterion_func(::AIC)
    return GLM.aic
end
function _regression_criterion_func(::AICC)
    return GLM.aicc
end
function _regression_criterion_func(::BIC)
    return GLM.bic
end
function _regression_criterion_func(::RSq)
    return GLM.r2
end
function _regression_criterion_func(::AdjRSq)
    return GLM.adjr2
end
function _regression_threshold(::AIC)
    return Inf
end
function _regression_threshold(::AICC)
    return Inf
end
function _regression_threshold(::BIC)
    return Inf
end
function _regression_threshold(::RSq)
    return -Inf
end
function _regression_threshold(::AdjRSq)
    return -Inf
end
function _get_forward_reg_incl_excl!(::MinValRegressionCriteria, value, excluded, included,
                                     threshold)
    val, key = findmin(value)
    idx = findfirst(x -> x == key, excluded)
    if val < threshold
        push!(included, popat!(excluded, idx))
        threshold = val
    end
    return threshold
end
function _get_forward_reg_incl_excl!(::MaxValRegressionCriteria, value, excluded, included,
                                     threshold)
    val, key = findmax(value)
    idx = findfirst(x -> x == key, excluded)
    if val > threshold
        push!(included, popat!(excluded, idx))
        threshold = val
    end
    return threshold
end
function _regression(::FReg, criterion::RegressionCriteria, x::DataFrame, y::AbstractVector)
    ovec = ones(length(y))
    namesx = names(x)

    criterion_func = _regression_criterion_func(criterion)
    threshold = _regression_threshold(criterion)

    included = String[]
    excluded = namesx
    for _ ∈ eachindex(y)
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

        threshold = _get_forward_reg_incl_excl!(criterion, value, excluded, included,
                                                threshold)

        if ni == length(excluded)
            break
        end
    end

    return included
end
function _get_backward_reg_incl!(::MinValRegressionCriteria, value, included, threshold)
    val, idx = findmin(value)
    if val < threshold
        i = findfirst(x -> x == idx, included)
        popat!(included, i)
        threshold = val
    end
    return threshold
end
function _get_backward_reg_incl!(::MaxValRegressionCriteria, value, included, threshold)
    val, idx = findmax(value)
    if val > threshold
        i = findfirst(x -> x == idx, included)
        popat!(included, i)
        threshold = val
    end
    return threshold
end
function _regression(::BReg, criterion::RegressionCriteria, x::DataFrame, y::AbstractVector)
    ovec = ones(length(y))
    fit_result = lm([ovec Matrix(x)], y)

    included = names(x)

    criterion_func = _regression_criterion_func(criterion)
    threshold = criterion_func(fit_result)

    for _ ∈ eachindex(y)
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

        threshold = _get_backward_reg_incl!(criterion, value, included, threshold)

        if ni == length(included)
            break
        end
    end

    return included
end
function regression(method::StepwiseRegression, x::DataFrame, y::DataFrame)
    features = names(x)
    rows = ncol(y)
    cols = ncol(x) + 1

    N = nrow(y)
    ovec = ones(N)

    loadings = zeros(rows, cols)

    for i ∈ 1:rows
        included = _regression(method, method.criterion, x, y[!, i])

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
function loadings_matrix(x::DataFrame, y::DataFrame, method::RegressionType = FReg())
    return regression(method, x, y)
end
function risk_factors(x::DataFrame, y::DataFrame; factor_type::FactorType = FactorType(),
                      cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                      mu_type::MeanEstimator = MuSimple())
    B = factor_type.B

    if isnothing(B)
        B = regression(factor_type.method, x, y)
    end
    namesB = names(B)
    old_posdef = nothing
    x1 = if "const" ∈ namesB
        if hasproperty(cov_type, :posdef) && !isa(cov_type.posdef, NoPosdef)
            old_posdef = cov_type.posdef
            cov_type.posdef = NoPosdef()
        end
        [ones(nrow(y)) Matrix(x)]
    else
        Matrix(x)
    end
    B_mtx = Matrix(B[!, setdiff(namesB, ("tickers",))])

    f_cov, f_mu = _sigma_mu(x1, cov_type, mu_type)

    if !isnothing(old_posdef)
        cov_type.posdef = old_posdef
        f_cov2 = f_cov[2:end, 2:end]
        posdef_fix!(cov_type.posdef, f_cov2)
        f_cov[2:end, 2:end] .= f_cov2
    end

    returns = x1 * transpose(B_mtx)
    mu = B_mtx * f_mu

    sigma = if factor_type.error
        e = Matrix(y) - returns
        S_e = diagm(vec(if isnothing(factor_type.var_w)
                            var(factor_type.ve, e; dims = 1)
                        else
                            var(factor_type.ve, e, factor_type.var_w; dims = 1)
                        end))
        B_mtx * f_cov * transpose(B_mtx) + S_e
    else
        B_mtx * f_cov * transpose(B_mtx)
    end

    if hasproperty(cov_type, :posdef)
        posdef_fix!(cov_type.posdef, sigma)
    end

    return mu, sigma, returns, B
end
"""
```
_mu_cov_w(tau, omega, P, Pi, Q, rf, sigma, delta, T, N, opt, cov_type, cov_flag = true)
```

Internal function for computing the Black Litterman statistics as defined in [`black_litterman`](@ref). See .

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
  - `opt`: any valid instance of `opt` for .
  - `cov_type`: any valid value from .
  - `cov_flag`: whether the matrix is a covariance matrix or not.

# Outputs

  - `mu`: asset expected returns vector obtained via the Black-Litterman model.
  - `cov_mtx`: asset covariance matrix obtained via the Black-Litterman model.
  - `w`: asset weights obtained via the Black-Litterman model.
  - `Pi_`: equilibrium excess returns after being adjusted by the views.
"""
function _bl_mu_cov_w(tau, omega, P, Pi, Q, rf, sigma, delta, T, N, posdef, denoise, logo)
    inv_tau_sigma = (tau * sigma) \ I
    inv_omega = omega \ I
    Pi_ = ((inv_tau_sigma + transpose(P) * inv_omega * P) \ I) *
          (inv_tau_sigma * Pi + transpose(P) * inv_omega * Q)
    M = (inv_tau_sigma + transpose(P) * inv_omega * P) \ I

    mu = Pi_ .+ rf
    sigma = sigma + M

    posdef_fix!(posdef, sigma)
    denoise!(denoise, posdef, sigma, T / N)
    logo!(logo, posdef, sigma)

    w = ((delta * sigma) \ I) * Pi_

    return mu, sigma, w, Pi_
end
function _omega(P, tau_sigma)
    return Diagonal(P * tau_sigma * transpose(P))
end
function _Pi(eq, delta, sigma, w, mu, rf)
    return eq ? delta * sigma * w : mu .- rf
end
"""
```
black_litterman(bl::BLType, X::AbstractMatrix, P::AbstractMatrix,
                         Q::AbstractVector, w::AbstractVector;
                         cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                         mu_type::MeanEstimator = MuSimple())
```
"""
function black_litterman(bl::BLType, X::AbstractMatrix, P::AbstractMatrix,
                         Q::AbstractVector, w::AbstractVector;
                         cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                         mu_type::MeanEstimator = MuSimple())
    sigma, mu = _sigma_mu(X, cov_type, mu_type)

    T, N = size(X)

    tau = 1 / T
    omega = _omega(P, tau * sigma)
    Pi = _Pi(bl.eq, bl.delta, sigma, w, mu, bl.rf)

    mu, sigma, w = _bl_mu_cov_w(tau, omega, P, Pi, Q, bl.rf, sigma, bl.delta, T, N,
                                bl.posdef, bl.denoise, bl.logo)[1:3]

    return mu, sigma, w
end
function black_litterman(bl::BBLType, X::AbstractMatrix; F::AbstractMatrix,
                         B::AbstractMatrix, P_f::AbstractMatrix, Q_f::AbstractVector,
                         cov_type::PortfolioOptimiserCovCor = PortCovCor(),
                         mu_type::MeanEstimator = MuSimple(), kwargs...)
    f_sigma, f_mu = _sigma_mu(F, cov_type, mu_type)

    f_mu .-= bl.rf

    if bl.constant
        alpha = B[:, 1]
        B = B[:, 2:end]
    end

    tau = 1 / size(X, 1)

    sigma = B * f_sigma * transpose(B)

    if bl.error
        D = X - F * transpose(B)
        D = Diagonal(vec(if isnothing(bl.var_w)
                             var(bl.ve, D; dims = 1)
                         else
                             var(bl.ve, D, bl.var_w; dims = 1)
                         end))
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

    mu = Pi_bbl .+ bl.rf

    if bl.constant
        mu .+= alpha
    end

    w = ((bl.delta * sigma_bbl) \ I) * mu

    return mu, sigma_bbl, w
end
function black_litterman(bl::ABLType, X::AbstractMatrix; w::AbstractVector,
                         F::Union{AbstractMatrix, Nothing}    = nothing,
                         B::Union{AbstractMatrix, Nothing}    = nothing,
                         P::Union{AbstractMatrix, Nothing}    = nothing,
                         P_f::Union{AbstractMatrix, Nothing}  = nothing,
                         Q::Union{AbstractVector, Nothing}    = nothing,
                         Q_f::Union{AbstractVector, Nothing}  = nothing,
                         cov_type::PortfolioOptimiserCovCor   = PortCovCor(;),
                         mu_type::MeanEstimator               = MuSimple(;),
                         f_cov_type::PortfolioOptimiserCovCor = PortCovCor(;),
                         f_mu_type::MeanEstimator             = MuSimple(;))
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
        sigma, mu = _sigma_mu(X, cov_type, mu_type)
    end

    if all_factor_provided
        f_sigma, f_mu = _sigma_mu(F, f_cov_type, f_mu_type)
    end

    if all_factor_provided && bl.constant
        alpha = B[:, 1]
        B = B[:, 2:end]
    end

    T, N = size(X)

    tau = 1 / T

    if all_asset_provided && !all_factor_provided
        sigma_a = sigma
        P_a = P
        Q_a = Q
        omega_a = _omega(P_a, tau * sigma_a)
        Pi_a = _Pi(bl.eq, bl.delta, sigma_a, w, mu, bl.rf)
    elseif !all_asset_provided && all_factor_provided
        sigma_a = f_sigma
        P_a = P_f
        Q_a = Q_f
        omega_a = _omega(P_a, tau * sigma_a)
        Pi_a = _Pi(bl.eq, bl.delta, sigma_a * transpose(B), w, f_mu, bl.rf)
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

        Pi_a = _Pi(bl.eq, bl.delta, vcat(sigma, f_sigma * transpose(B)), w, vcat(mu, f_mu),
                   bl.rf)
    end

    mu_a, sigma_a, w_a, Pi_a_ = _bl_mu_cov_w(tau, omega_a, P_a, Pi_a, Q_a, bl.rf, sigma_a,
                                             bl.delta, T, N, bl.posdef, bl.denoise, bl.logo)

    if !all_asset_provided && all_factor_provided
        mu_a = B * mu_a
        sigma_a = B * sigma_a * transpose(B)
        posdef_fix!(bl.posdef, sigma_a)
        denoise!(bl.denoise, bl.posdef, sigma_a, T / N)
        logo!(bl.logo, bl.posdef, sigma_a)
        w_a = ((bl.delta * sigma_a) \ I) * B * Pi_a_
    end

    if all_factor_provided && bl.constant
        mu_a = mu_a[1:N] .+ alpha
    end

    return mu_a[1:N], sigma_a[1:N, 1:N], w_a[1:N]
end

"""
```julia
owa_gmd(T::Integer)
```

Computes the Gini Mean Difference (GMD) of a returns series [^OWA].

# Inputs

# Outputs

[^OWA]: [Cajas, Dany, OWA Portfolio Optimization: A Disciplined Convex Programming Framework (December 18, 2021). Available at SSRN: https://ssrn.com/abstract=3988927 or http://dx.doi.org/10.2139/ssrn.3988927](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3988927)
"""
function owa_gmd(T::Integer)
    w = Vector{typeof(inv(T))}(undef, T)
    for i ∈ 1:T
        w[i] = 2 * i - 1 - T
    end
    w = 2 / (T * (T - 1)) * w

    return w
end

"""
```julia
owa_cvar(T::Integer; alpha::Real = 0.05)
```

Calculate the OWA weights corresponding to the Critical Value at Risk (CVaR) of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_cvar(T::Integer, alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))

    k = floor(Int, T * alpha)
    w = zeros(typeof(alpha), T)
    w[1:k] .= -1 / (T * alpha)
    w[k + 1] = -1 - sum(w[1:k])

    return w
end

"""
```julia
owa_wcvar(T::Integer, alphas::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
```

Compute the OWA weights for the Weighted Conditional Value at Risk (WCVaR) of a returns series [^OWA].

# Inputs

  - `alphas`: `N×1` vector of significance levels of each CVaR model, where `N` is the number of models, each .
  - `weights`: `N×1` vector of weights of each CVaR model, where `N` is the number of models.

# Outputs
"""
function owa_wcvar(T::Integer, alphas::AbstractVector{<:Real},
                   weights::AbstractVector{<:Real})
    w = zeros(promote_type(eltype(alphas), eltype(weights)), T)
    for (i, j) ∈ zip(alphas, weights)
        w .+= owa_cvar(T, i) * j
    end

    return w
end

"""
```julia
owa_tg(T::Integer; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100)
```

Compute the OWA weights for the Tail Gini of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_tg(T::Integer; alpha_i::Real = 1e-4, alpha::Real = 0.05, a_sim::Integer = 100)
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))

    alphas = range(; start = alpha_i, stop = alpha, length = a_sim)
    n = length(alphas)
    w = Vector{typeof(alpha)}(undef, n)

    w[1] = alphas[2] * alphas[1] / alphas[n]^2
    for i ∈ 2:(n - 1)
        w[i] = (alphas[i + 1] - alphas[i - 1]) * alphas[i] / alphas[n]^2
    end
    w[n] = (alphas[n] - alphas[n - 1]) / alphas[n]

    w = owa_wcvar(T, alphas, w)

    return w
end

"""
```julia
owa_wr(T::Integer)
```

Compute the OWA weights for the Worst Realisation (WR) of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_wr(T::Integer)
    w = zeros(typeof(inv(T)), T)
    w[1] = -1

    return w
end

"""
```julia
owa_rg(T::Integer)
```

Compute the OWA weights for the Range of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_rg(T::Integer)
    w = zeros(typeof(inv(T)), T)
    w[1] = -1
    w[T] = 1
    return w
end

"""
```julia
owa_rcvar(T::Integer; alpha::Real = 0.05, beta::Real = alpha)
```

Compute the OWA weights for the CVaR Range of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_rcvar(T::Integer; alpha::Real = 0.05, beta::Real = alpha)
    w = owa_cvar(T, alpha) .- reverse(owa_cvar(T, beta))
    return w
end

"""
```julia
owa_rwcvar(T::Integer, alphas::AbstractVector{<:Real}, weights_a::AbstractVector{<:Real};
           betas::AbstractVector{<:Real} = alphas,
           weights_b::AbstractVector{<:Real} = weights_b)
```

Compute the OWA weights for the Weighted Conditional Value at Risk (WCVaR) of a returns series [^OWA].

# Inputs

  - `alphas`: `N×1` vector of significance levels of the losses for each CVaR model, where `N` is the number of losses models, each .
  - `weights_a`: `N×1` vector of weights of the losses for each CVaR model, where `N` is the number of losses models.
  - `betas`: `M×1` vector of significance levels of the gains for each CVaR model, where `M` is the number of gains models, each .
  - `weights_b`: `M×1` vector of weights of the gains for each CVaR model, where `M` is the number of gains models.

# Outputs
"""
function owa_rwcvar(T::Integer, alphas::AbstractVector{<:Real},
                    weights_a::AbstractVector{<:Real},
                    betas::AbstractVector{<:Real} = alphas,
                    weights_b::AbstractVector{<:Real} = weights_a)
    w = owa_wcvar(T, alphas, weights_a) .- reverse(owa_wcvar(T, betas, weights_b))

    return w
end

"""
```julia
owa_rtg(T::Integer; alpha_i::Real = 0.0001, alpha::Real = 0.05, a_sim::Integer = 100,
        beta_i::Real = alpha_i, beta::Real = alpha, b_sim::Integer = a_sim)
```

Compute the OWA weights for the Tail Gini Range of a returns series [^OWA].

# Inputs

# Outputs
"""
function owa_rtg(T::Integer; alpha_i::Real = 0.0001, alpha::Real = 0.05,
                 a_sim::Integer = 100, beta_i::Real = alpha_i, beta::Real = alpha,
                 b_sim::Integer = a_sim)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) .-
        reverse(owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim))

    return w
end

"""
```julia
_optimise_JuMP_model(model, solvers)
```

Internal function to optimise an OWA JuMP model.

# Inputs

  - `model`: JuMP model.

# Outputs

  - `term_status`: JuMP termination status.

  - `solvers_tried`: Dictionary that contains a dictionary of failed optimisations. `Dict(key => Dict(...))`, where `key` is the solver key used for the iteration of `solver` that failed.

      + If an MOI call fails on a model:

          * `Dict(:jump_error => jump_error)`: [`JuMP`](https://jump.dev/JuMP.jl/stable/moi/reference/errors/) error code.

      + If the optimiser fails to optimise the model satisfactorily:

          * `Dict(:objective_val => JuMP.objective_value(model), :term_status => term_status, :params => haskey(val, :params) ? val[:params] : missing)`, where `val` is the value of the dictionary corresponding to `key`.
"""
function _optimise_JuMP_model(model, solvers)
    solvers_tried = Dict()

    sucess = false
    for (key, val) ∈ solvers
        if haskey(val, :solver)
            set_optimizer(model, val[:solver])
        end

        if haskey(val, :params)
            for (attribute, value) ∈ val[:params]
                set_attribute(model, attribute, value)
            end
        end

        if haskey(val, :check_sol)
            check_sol = val[:check_sol]
        else
            check_sol = (;)
        end

        try
            JuMP.optimize!(model)
        catch jump_error
            push!(solvers_tried, key => Dict(:jump_error => jump_error))
            continue
        end

        if is_solved_and_feasible(model; check_sol...)
            sucess = true
            break
        else
            term_status = termination_status(model)
        end

        push!(solvers_tried,
              key => Dict(:objective_val => objective_value(model),
                          :term_status => term_status,
                          :params => haskey(val, :params) ? val[:params] : missing))
    end

    return sucess, solvers_tried
end

"""
```julia
_crra_method(weights::AbstractMatrix{<:Real}, k::Integer, g::Real)
```

Internal function for computing the Normalized Constant Relative Risk Aversion coefficients.

# Inputs

  - `weights`: `T×(k-1)` matrix where T is the number of observations and `k` the order of the L-moments to combine, the `i`'th column contains the weights for the `(i+1)`'th L-moment.
  - `k`: the maximum order of the L-moments.
  - `g`: the risk aversion coefficient.

# Outputs

  - `w`: `T×1` ordered weight vector of the combined L-moments.
"""
function _crra_method(weights::AbstractMatrix{<:Real}, k::Integer, g::Real)
    phis = Vector{eltype(weights)}(undef, k - 1)
    e = 1
    for i ∈ 1:(k - 1)
        e *= g + i - 1
        phis[i] = e / factorial(i + 1)
    end

    phis ./= sum(phis)
    a = weights * phis

    w = similar(a)
    w[1] = a[1]
    for i ∈ 2:length(a)
        w[i] = maximum(a[1:i])
    end

    return w
end

"""
```julia
owa_l_moment(T::Integer; k::Integer = 2)
```

Calculates the OWA weights of the k'th linear moment (L-moment) of a returns series [OWAL](@cite).

# Inputs

  - `k`: order of the L-moment.

# Outputs
"""
function owa_l_moment(T::Integer, k::Integer = 2)
    w = Vector{typeof(inv(T * k))}(undef, T)
    T, k = promote(T, k)
    for i ∈ 1:T
        a = zero(k)
        for j ∈ 0:(k - 1)
            a += (-1)^j *
                 binomial(k - 1, j) *
                 binomial(i - 1, k - 1 - j) *
                 binomial(T - i, j)
        end
        a *= 1 / (k * binomial(T, k))
        w[i] = a
    end

    return w
end

"""
```julia
owa_l_moment_crm(T::Integer; k::Integer = 2, method::Symbol = :SD, g::Real = 0.5,
                 max_phi::Real = 0.5, solvers = Dict())
```

Compute the OWA weights for the convex risk measure considering higher order L-moments [OWAL](@cite).

# Inputs

  - `k`: order of the L-moment, `k ≥ 2`.

  - `method`: method for computing the weights used to combine L-moments higher than 2, used in [`OWAMethods`](@ref).

      + `:CRRA:` Normalised Constant Relative Risk Aversion Coefficients.
      + `:E`: Maximum Entropy. Solver must support `MOI.RelativeEntropyCone` and `MOI.NormOneCone`.
      + `:SS`: Minimum Sum of Squares. Solver must support `MOI.SecondOrderCone`.
      + `:SD`: Minimum Square Distance. Solver must support `MOI.SecondOrderCone`.
  - `g`: the risk aversion coefficient.
  - `max_phi`: maximum weight constraint of the L-moments.

# Outputs
"""
function _owa_l_moment_crm(method::CRRA, ::Any, k, weights, ::Any)
    return _crra_method(weights, k, method.g)
end
function _owa_model_setup(method, T, weights)
    n = size(weights, 2)
    model = JuMP.Model()
    max_phi = method.max_phi
    @variable(model, theta[1:T])
    @variable(model, 0 .<= phi[1:n] .<= max_phi)
    @constraint(model, sum(phi) == 1)
    @constraint(model, theta .== weights * phi)
    @constraint(model, phi[2:end] .<= phi[1:(end - 1)])
    @constraint(model, theta[2:end] .>= theta[1:(end - 1)])
    return model
end
function _owa_model_solve(model, weights, solvers, k)
    success, solvers_tried = _optimise_JuMP_model(model, solvers)
    return if success
        phi = model[:phi]
        phis = value.(phi)
        phis ./= sum(phis)
        w = weights * phis
    else
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.owa_l_moment_crm))"
        @warn("$funcname: model could not be optimised satisfactorily.\nMethod: $method\nSolvers: $solvers_tried.\nReverting to crra method.")
        w = _crra_method(weights, k, 0.5)
    end
end
function _owa_l_moment_crm(method::MaxEntropy, T, k, weights, solvers)
    model = _owa_model_setup(method, T, weights)
    @variable(model, t)
    @variable(model, x[1:T])
    @constraint(model, sum(x) == 1)
    @constraint(model, [t; ones(T); x] ∈ MOI.RelativeEntropyCone(2 * T + 1))
    theta = model[:theta]
    @constraint(model, [i = 1:T], [x[i]; theta[i]] ∈ MOI.NormOneCone(2))
    @objective(model, Max, -t)
    return _owa_model_solve(model, weights, solvers, k)
end
function _owa_l_moment_crm(method::MinSumSq, T, k, weights, solvers)
    model = _owa_model_setup(method, T, weights)
    @variable(model, t)
    theta = model[:theta]
    @constraint(model, [t; theta] ∈ SecondOrderCone())
    @objective(model, Min, t)
    return _owa_model_solve(model, weights, solvers, k)
end
function _owa_l_moment_crm(method::MinSqDist, T, k, weights, solvers)
    model = _owa_model_setup(method, T, weights)
    @variable(model, t)
    theta = model[:theta]
    @expression(model, theta_diff, theta[2:end] .- theta[1:(end - 1)])
    @constraint(model, [t; theta_diff] ∈ SecondOrderCone())
    @objective(model, Min, t)
    return _owa_model_solve(model, weights, solvers, k)
end
function owa_l_moment_crm(T::Integer; k::Integer = 2, method::OWAMethods = MinSqDist(),
                          solvers = Dict())
    @smart_assert(k >= 2)
    rg = 2:k
    weights = Matrix{typeof(inv(T * k))}(undef, T, length(rg))
    for i ∈ rg
        wi = (-1)^i * owa_l_moment(T, i)
        weights[:, i - 1] .= wi
    end
    return _owa_l_moment_crm(method, T, k, weights, solvers)
end

"""
```
is_leaf(a::ClusterNode)
```
"""
function is_leaf(a::ClusterNode)
    return isnothing(a.left)
end

"""
```
pre_order(a::ClusterNode, func::Function = x -> x.id)
```
"""
function pre_order(a::ClusterNode, func::Function = x -> x.id)
    n = a.level
    curNode = Vector{ClusterNode}(undef, 2 * n)
    lvisited = Set()
    rvisited = Set()
    curNode[1] = a
    k = 1
    preorder = Int[]

    while k >= 1
        nd = curNode[k]
        ndid = nd.id
        if is_leaf(nd)
            push!(preorder, func(nd))
            k = k - 1
        else
            if ndid ∉ lvisited
                curNode[k + 1] = nd.left
                push!(lvisited, ndid)
                k = k + 1
            elseif ndid ∉ rvisited
                curNode[k + 1] = nd.right
                push!(rvisited, ndid)
                k = k + 1
                # If we've visited the left and right of this non-leaf
                # node already, go up in the tree.
            else
                k = k - 1
            end
        end
    end

    return preorder
end

"""
```
to_tree(a::Hclust)
```
"""
function to_tree(a::Hclust)
    n = length(a.order)
    d = Vector{ClusterNode}(undef, 2 * n - 1)
    for i ∈ 1:n
        d[i] = ClusterNode(i)
    end
    merges = a.merges
    heights = a.heights
    nd = nothing

    for (i, height) ∈ pairs(heights)
        fi = merges[i, 1]
        fj = merges[i, 2]

        fi = fi < 0 ? -fi : fi + n
        fj = fj < 0 ? -fj : fj + n

        nd = ClusterNode(i + n, d[fi], d[fj], height)
        d[n + i] = nd
    end
    return nd, d
end
function _two_diff_gap_stat(dist, clustering, max_k = 0)
    N = size(dist, 1)
    cluster_lvls = [cutree(clustering; k = i) for i ∈ 1:N]

    if iszero(max_k)
        max_k = ceil(Int, sqrt(size(dist, 1)))
    end

    c1 = min(N, max_k)
    W_list = Vector{eltype(dist)}(undef, c1)

    for i ∈ 1:c1
        lvl = cluster_lvls[i]
        c2 = maximum(unique(lvl))
        D_list = Vector{eltype(dist)}(undef, c2)
        for j ∈ 1:c2
            cluster = findall(lvl .== j)
            cluster_dist = dist[cluster, cluster]
            if isempty(cluster_dist)
                continue
            end
            M = size(cluster_dist, 1)
            C_list = Vector{eltype(dist)}(undef, Int(M * (M - 1) / 2))
            k = 1
            for col ∈ 1:M
                for row ∈ (col + 1):M
                    C_list[k] = cluster_dist[row, col]
                    k += 1
                end
            end
            D_list[j] = k == 1 ? zero(eltype(dist)) : std(C_list; corrected = false)
        end
        W_list[i] = sum(D_list)
    end

    gaps = fill(-Inf, c1)

    if c1 > 2
        gaps[1:(end - 2)] .= W_list[1:(end - 2)] .+ W_list[3:end] .- 2 * W_list[2:(end - 1)]
    end

    k = all(isinf.(gaps)) ? length(gaps) : k = argmax(gaps)

    return k
end
function _calc_k_clusters(::TwoDiff, dist::AbstractMatrix, clustering, max_k::Integer)
    return _two_diff_gap_stat(dist, clustering, max_k)
end
function _std_silhouette_score(dist, clustering, max_k = 0, metric = nothing)
    N = size(dist, 1)
    cluster_lvls = [cutree(clustering; k = i) for i ∈ 1:N]

    if iszero(max_k)
        max_k = ceil(Int, sqrt(size(dist, 1)))
    end

    c1 = min(N, max_k)
    W_list = Vector{eltype(dist)}(undef, c1)
    W_list[1] = -Inf
    for i ∈ 2:c1
        lvl = cluster_lvls[i]
        sl = silhouettes(lvl, dist; metric = metric)
        msl = mean(sl)
        W_list[i] = msl / std(sl; mean = msl)
    end

    k = all(.!isfinite.(W_list)) ? length(W_list) : k = argmax(W_list)

    return k
end
function _calc_k_clusters(method::StdSilhouette, dist::AbstractMatrix, clustering,
                          max_k::Integer)
    return _std_silhouette_score(dist, clustering, max_k, method.metric)
end
"""
```
calc_k_clusters(hclust_opt::HCOpt, dist::AbstractMatrix, clustering)
```
"""
function calc_k_clusters(hclust_opt::HCOpt, dist::AbstractMatrix, clustering)
    if !iszero(hclust_opt.k)
        return hclust_opt.k
    end
    return _calc_k_clusters(hclust_opt.k_method, dist, clustering, hclust_opt.max_k)
end

function _hcluster(ca::HAC, X::AbstractMatrix,
                   cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                   dist_type::DistanceMethod = DistanceCanonical(),
                   hclust_opt::HCOpt = HCOpt())
    dist_type = _get_default_dist(dist_type, cor_type)
    if hasproperty(cor_type.ce, :absolute) && hasproperty(dist_type, :absolute)
        dist_type.absolute = cor_type.ce.absolute
    end

    S = cor(cor_type, X)
    D = dist(dist_type, S, X)

    clustering = hclust(D; linkage = ca.linkage, branchorder = hclust_opt.branchorder)
    k = calc_k_clusters(hclust_opt, D, clustering)

    return clustering, k, S, D
end
function _hcluster(ca::DBHT, X::AbstractMatrix,
                   cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                   dist_type::DistanceMethod = DistanceCanonical(),
                   hclust_opt::HCOpt = HCOpt())
    dist_type = _get_default_dist(dist_type, cor_type)
    if hasproperty(cor_type.ce, :absolute) && hasproperty(dist_type, :absolute)
        dist_type.absolute = cor_type.ce.absolute
    end

    S = cor(cor_type, X)
    D = dist(dist_type, S, X)
    S = dbht_similarity(ca.similarity, S, D)

    clustering = DBHTs(D, S; branchorder = hclust_opt.branchorder, method = ca.root_method)[end]
    k = calc_k_clusters(hclust_opt, D, clustering)

    return clustering, k, S, D
end
function cluster_assets(X::AbstractMatrix;
                        cor_type::PortfolioOptimiserCovCor = PortCovCor(),
                        dist_type::DistanceMethod = DistanceCanonical(),
                        hclust_alg::HClustAlg = HAC(), hclust_opt::HCOpt = HCOpt())
    clustering, k, S, D = _hcluster(hclust_alg, X, cor_type, dist_type, hclust_opt)

    idx = cutree(clustering; k = k)

    return idx, clustering, k, S, D
end

export cov_returns, posdef_fix!, dbht_similarity, dist, calc_num_bins, calc_hist_data,
       mutual_info, variation_info, cor_distance, cov_distance, lower_tail_dependence,
       cov_gerber, duplication_matrix, elimination_matrix, summation_matrix,
       dup_elim_sum_matrices, errPDF, find_max_eval, denoise!, target_mean, logo!, cokurt,
       coskew, gen_bootstrap, vec_of_vecs_to_mtx, calc_sets, commutation_matrix, calc_k_wc,
       prep_dim_red_reg, regression, loadings_matrix, risk_factors, black_litterman,
       owa_gmd, owa_cvar, owa_wcvar, owa_tg, owa_wr, owa_rg, owa_rcvar, owa_rwcvar, owa_rtg,
       owa_l_moment, owa_l_moment_crm, is_leaf, pre_order, to_tree, calc_k_clusters,
       cluster_assets
