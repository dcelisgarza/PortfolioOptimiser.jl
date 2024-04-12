abstract type AbstractBins end
struct KN <: AbstractBins end
struct FD <: AbstractBins end
struct SC <: AbstractBins end
const AstroBins = Union{KN, FD, SC}
struct HGR <: AbstractBins end

mutable struct CorMutualInfo <: StatsBase.CovarianceEstimator
    bins::Union{<:Integer, <:AbstractBins}
    normalise::Bool
end
function CorMutualInfo(; bins::Union{<:Integer, <:AbstractBins} = HGR(),
                       normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return CorMutualInfo(bins, normalise)
end
function Base.setproperty!(obj::CorMutualInfo, sym::Symbol, val)
    if sym == :bins
        if isa(val, Integer)
            @smart_assert(val > zero(val))
        end
    end
    return setfield!(obj, sym, val)
end
function _bin_width_func(::KN)
    return pyimport("astropy.stats").knuth_bin_width
end
function _bin_width_func(::FD)
    return pyimport("astropy.stats").freedman_bin_width
end
function _bin_width_func(::SC)
    return pyimport("astropy.stats").scott_bin_width
end
function _bin_width_func(::Union{HGR, <:Integer})
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
function calc_num_bins(::HGR, xj, xi, j, i, bin_width_func, N)
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
                               bins::Union{<:AbstractBins, <:Integer} = KN(),
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
function mutual_info(X::AbstractMatrix, bins::Union{<:AbstractBins, <:Integer} = KN(),
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
function variation_info(X::AbstractMatrix, bins::Union{<:AbstractBins, <:Integer} = KN(),
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
function StatsBase.cor(ce::CorMutualInfo, X::AbstractMatrix, args...; kwargs...)
    return mutual_info(X, ce.bins, ce.normalise)
end
function dist(ce::CorMutualInfo, X::AbstractMatrix, args...; kwargs...)
    return variation_info(X, ce.bins, ce.normalise)
end
@kwdef mutable struct CorDistance <: StatsBase.CovarianceEstimator
    distance::Distances.UnionMetric = Distances.Euclidean()
    args::Tuple = ()
    kwargs::NamedTuple = (;)
end
function cor_distance(ce::CorDistance, v1::AbstractVector, v2::AbstractVector)
    N = length(v1)
    @smart_assert(N == length(v2) && N > 1)

    N2 = N^2

    a = pairwise(ce.distance, v1, ce.args...; ce.kwargs...)
    b = pairwise(ce.distance, v2, ce.args...; ce.kwargs...)
    A = a .- mean(a; dims = 1) .- mean(a; dims = 2) .+ mean(a)
    B = b .- mean(b; dims = 1) .- mean(b; dims = 2) .+ mean(b)

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
function StatsBase.cor(ce::CorDistance, X::AbstractMatrix, args...; kwargs...)
    return cor_distance(ce::CorDistance, X::AbstractMatrix)
end
mutable struct CorLowerTailDependence <: StatsBase.CovarianceEstimator
    alpha::Real
end
function CorLowerTailDependence(; alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return CorLowerTailDependence(alpha)
end
function Base.setproperty!(obj::CorLowerTailDependence, sym::Symbol, val)
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
function StatsBase.cor(ce::CorLowerTailDependence, X::AbstractMatrix, args...; kwargs...)
    return lower_tail_dependence(X, ce.alpha)
end
abstract type PosdefFix end
@kwdef mutable struct Nearest <: PosdefFix
    method::NearestCorrelationMatrix.NCMAlgorithm = NearestCorrelationMatrix.Newton(;
                                                                                    tau = 1e-12)
end
"""
```
_posdef_fix!(method::Nearest, X::AbstractMatrix)
```

Overload this for other posdef fix methods.
"""
function _posdef_fix!(method::Nearest, X::AbstractMatrix)
    NCM.nearest_cor!(X, method)
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

abstract type CorGerber <: StatsBase.CovarianceEstimator end
abstract type CorGerberBasic <: CorGerber end
abstract type CorSB <: CorGerber end
abstract type CorGerberSB <: CorGerber end
mutable struct CorGerber0{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    mean_args::Tuple
    mean_kwargs::NamedTuple
    std_args::Tuple
    std_kwargs::NamedTuple
    posdef::PosdefFix
end
function CorGerber0(; normalise::Bool = false, threshold::Real = 0.5, mean_args::Tuple = (),
                    mean_kwargs::NamedTuple = (; dims = 1), std_args::Tuple = (),
                    std_kwargs::NamedTuple = (; dims = 1), posdef::PosdefFix = Nearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber0{typeof(threshold)}(normalise, threshold, mean_args, mean_kwargs,
                                         std_args, std_kwargs, posdef)
end
mutable struct CorGerber1{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    mean_args::Tuple
    mean_kwargs::NamedTuple
    std_args::Tuple
    std_kwargs::NamedTuple
    posdef::PosdefFix
end
function CorGerber1(; normalise::Bool = false, threshold::Real = 0.5, mean_args::Tuple = (),
                    mean_kwargs::NamedTuple = (; dims = 1), std_args::Tuple = (),
                    std_kwargs::NamedTuple = (; dims = 1), posdef::PosdefFix = Nearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber1{typeof(threshold)}(normalise, threshold, mean_args, mean_kwargs,
                                         std_args, std_kwargs, posdef)
end
mutable struct CorGerber2{T1 <: Real} <: CorGerberBasic
    normalise::Bool
    threshold::T1
    mean_args::Tuple
    mean_kwargs::NamedTuple
    std_args::Tuple
    std_kwargs::NamedTuple
    posdef::PosdefFix
end
function CorGerber2(; normalise::Bool = false, threshold::Real = 0.5, mean_args::Tuple = (),
                    mean_kwargs::NamedTuple = (; dims = 1), std_args::Tuple = (),
                    std_kwargs::NamedTuple = (; dims = 1), posdef::PosdefFix = Nearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return CorGerber2{typeof(threshold)}(normalise, threshold, mean_args, mean_kwargs,
                                         std_args, std_kwargs, posdef)
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
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _cor_gerber(ce::CorGerber0, X::AbstractMatrix, std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _cor_gerber_norm(ce::CorGerber1, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _cor_gerber(ce::CorGerber1, X::AbstractMatrix, std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
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

    sigma = H ./ (h * transpose(h))
    posdef_fix!(ce.posdef, sigma)

    return sigma
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

    sigma = H ./ (h * transpose(h))
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
mutable struct CorSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    mean_args::Tuple
    mean_kwargs::NamedTuple
    std_args::Tuple
    std_kwargs::NamedTuple
    posdef::PosdefFix
end
function CorSB0(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4, n::Real = 2, mean_args::Tuple = (),
                mean_kwargs::NamedTuple = (; dims = 1), std_args::Tuple = (),
                std_kwargs::NamedTuple = (; dims = 1), posdef::PosdefFix = Nearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorSB0{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                    threshold,
                                                                                    c1, c2,
                                                                                    c3, n,
                                                                                    mean_args,
                                                                                    mean_kwargs,
                                                                                    std_args,
                                                                                    std_kwargs,
                                                                                    posdef)
end
mutable struct CorSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    mean_args::Tuple
    mean_kwargs::NamedTuple
    std_args::Tuple
    std_kwargs::NamedTuple
    posdef::PosdefFix
end
function CorSB1(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                c2::Real = 0.5, c3::Real = 4, n::Real = 2, mean_args::Tuple = (),
                mean_kwargs::NamedTuple = (; dims = 1), std_args::Tuple = (),
                std_kwargs::NamedTuple = (; dims = 1), posdef::PosdefFix = Nearest(;))
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2))
    @smart_assert(c3 > c2)
    return CorSB1{typeof(threshold), typeof(c1), typeof(c2), typeof(c3), typeof(n)}(normalise,
                                                                                    threshold,
                                                                                    c1, c2,
                                                                                    c3, n,
                                                                                    mean_args,
                                                                                    mean_kwargs,
                                                                                    std_args,
                                                                                    std_kwargs,
                                                                                    posdef)
end
mutable struct CorGerberSB0{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    mean_args::Tuple
    mean_kwargs::NamedTuple
    std_args::Tuple
    std_kwargs::NamedTuple
    posdef::PosdefFix
end
function CorGerberSB0(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4, n::Real = 2, mean_args::Tuple = (),
                      mean_kwargs::NamedTuple = (; dims = 1), std_args::Tuple = (),
                      std_kwargs::NamedTuple = (; dims = 1), posdef::PosdefFix = Nearest(;))
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
                                                                                          mean_args,
                                                                                          mean_kwargs,
                                                                                          std_args,
                                                                                          std_kwargs,
                                                                                          posdef)
end
mutable struct CorGerberSB1{T1, T2, T3, T4, T5} <: CorSB
    normalise::Bool
    threshold::T1
    c1::T2
    c2::T3
    c3::T4
    n::T5
    mean_args::Tuple
    mean_kwargs::NamedTuple
    std_args::Tuple
    std_kwargs::NamedTuple
    posdef::PosdefFix
end
function CorGerberSB1(; normalise::Bool = false, threshold::Real = 0.5, c1::Real = 0.5,
                      c2::Real = 0.5, c3::Real = 4, n::Real = 2, mean_args::Tuple = (),
                      mean_kwargs::NamedTuple = (; dims = 1), std_args::Tuple = (),
                      std_kwargs::NamedTuple = (; dims = 1), posdef::PosdefFix = Nearest(;))
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
                                                                                          mean_args,
                                                                                          mean_kwargs,
                                                                                          std_args,
                                                                                          std_kwargs,
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
=#
function _cor_gerber_norm(ce::CorSB0, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _cor_gerber(ce::CorSB0, X::AbstractMatrix, mean_vec::AbstractVector,
                     std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _cor_gerber_norm(ce::CorSB1, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _cor_gerber(ce::CorSB1, X::AbstractMatrix, mean_vec::AbstractVector,
                     std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _cor_gerber_norm(ce::CorGerberSB0, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _cor_gerber(ce::CorGerberSB0, X::AbstractMatrix, mean_vec::AbstractVector,
                     std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _cor_gerber_norm(ce::CorGerberSB1, X::AbstractMatrix, mean_vec::AbstractVector,
                          std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _cor_gerber(ce::CorGerberSB1, X::AbstractMatrix, mean_vec::AbstractVector,
                     std_vec::AbstractVector)
    T, N = size(X)
    sigma = Matrix{eltype(X)}(undef, N, N)
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
            sigma[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end

    sigma .= Symmetric(sigma, :U)
    posdef_fix!(ce.posdef, sigma)

    return sigma
end
function _gerber(ce::CorGerberBasic, X::AbstractMatrix, std_vec::AbstractVector)
    return if ce.normalise
        mean_vec = vec(mean(X, ce.mean_args...; ce.mean_kwargs...))
        _cor_gerber_norm(ce, X, mean_vec, std_vec)
    else
        _cor_gerber(ce, X, std_vec)
    end
end
function _gerber(ce::Union{CorSB, CorGerberSB}, X::AbstractMatrix, std_vec::AbstractVector)
    mean_vec = vec(mean(X, ce.mean_args...; ce.mean_kwargs...))
    return if ce.normalise
        _cor_gerber_norm(ce, X, mean_vec, std_vec)
    else
        _cor_gerber(ce, X, mean_vec, std_vec)
    end
end
function cor_gerber(ce::CorGerber, X::AbstractMatrix)
    std_vec = vec(std(X, ce.std_args...; ce.std_kwargs...))
    return Symmetric(_gerber(ce, X, std_vec))
end
function cov_gerber(ce::CorGerber, X::AbstractMatrix)
    std_vec = vec(std(X, ce.std_args...; ce.std_kwargs...))
    return Symmetric(_gerber(ce, X, std_vec) .* (std_vec * transpose(std_vec)))
end
function StatsBase.cor(ce::CorGerber, X::AbstractMatrix, args...; kwargs...)
    return cor_gerber(ce, X)
end
function StatsBase.cov(ce::CorGerber, X::AbstractMatrix, args...; kwargs...)
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

### Tested. Move on to denoising.

abstract type CorMetric <: Distances.UnionMetric end
struct MLPDistance <: CorMetric end