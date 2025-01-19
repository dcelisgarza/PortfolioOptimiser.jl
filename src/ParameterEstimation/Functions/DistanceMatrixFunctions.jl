"""
    dist(de::GenDistMLP, X::AbstractMatrix, ::Any)
"""
function dist(de::GenDistMLP, X::AbstractMatrix, ::Any)
    return Symmetric(sqrt.(if !de.absolute
                               power = de.power
                               scaler = isodd(power) ? 0.5 : 1.0
                               clamp!((one(eltype(X)) .- X .^ power) * scaler,
                                      zero(eltype(X)), one(eltype(X)))
                           else
                               power = de.power
                               clamp!(one(eltype(X)) .- abs.(X) .^ power, zero(eltype(X)),
                                      one(eltype(X)))
                           end))
end
function dist(de::GenDistDistMLP, X::AbstractMatrix, ::Any)
    _X = Symmetric(sqrt.(if !de.absolute
                             power = de.power
                             scaler = isodd(power) ? 0.5 : 1.0
                             clamp!((one(eltype(X)) .- X .^ power) * scaler,
                                    zero(eltype(X)), one(eltype(X)))
                         else
                             power = de.power
                             clamp!(one(eltype(X)) .- abs.(X) .^ power, zero(eltype(X)),
                                    one(eltype(X)))
                         end))

    return Symmetric(Distances.pairwise(de.distance, _X, de.args...; de.kwargs...))
end
function dist(de::DistMLP, X::AbstractMatrix, ::Any)
    return Symmetric(sqrt.(if !de.absolute
                               clamp!((one(eltype(X)) .- X) / 2, zero(eltype(X)),
                                      one(eltype(X)))
                           else
                               clamp!(one(eltype(X)) .- X, zero(eltype(X)), one(eltype(X)))
                           end))
end
function dist(de::DistDistMLP, X::AbstractMatrix, ::Any)
    _X = sqrt.(if !de.absolute
                   clamp!((one(eltype(X)) .- X) / 2, zero(eltype(X)), one(eltype(X)))
               else
                   clamp!(one(eltype(X)) .- X, zero(eltype(X)), one(eltype(X)))
               end)

    return Symmetric(Distances.pairwise(de.distance, _X, de.args...; de.kwargs...))
end
function dist(::DistLog, X::AbstractMatrix, ::Any)
    return Symmetric(-log.(X))
end
function dist(de::DistDistLog, X::AbstractMatrix, ::Any)
    D = dist(DistLog(), X, nothing)
    return Symmetric(Distances.pairwise(de.distance, D, de.args...; de.kwargs...))
end
function dist(de::DistVarInfo, ::Any, Y::AbstractMatrix)
    return variation_info(Y, de.bins, de.normalise)
end
function dist(de::DistDistVarInfo, ::Any, Y::AbstractMatrix)
    D = dist(de.de, nothing, Y)
    return Symmetric(Distances.pairwise(de.distance, D, de.args...; de.kwargs...))
end
function dist(::DistCor, X::AbstractMatrix, ::Any)
    return Symmetric(sqrt.(clamp!(one(eltype(X)) .- X, zero(eltype(X)), one(eltype(X)))))
end
function dist(de::DistDistCor, X::AbstractMatrix, ::Any)
    D = dist(DistCor(), X, nothing)
    return Symmetric(Distances.pairwise(de.distance, D, de.args...; de.kwargs...))
end
function set_absolute_dist!(dist_type::AbsoluteDist, cor_type::PortCovCor)
    return set_absolute_dist!(dist_type, cor_type.ce)
end
function set_absolute_dist!(dist_type::AbsoluteDist, cor_type::AbsoluteCovCor)
    overwrite = dist_type.overwrite
    if overwrite
        dist_type.absolute = cor_type.absolute
    end
    return nothing
end
function set_absolute_dist!(args...)
    return nothing
end
"""
    default_dist(dist_type::DistCanonical, cor_type::PortCovCor)
"""
function default_dist(dist_type::DistCanonical, cor_type::PortCovCor)
    return default_dist(dist_type, cor_type.ce)
end
function default_dist(::DistCanonical, cor_type::CovMutualInfo)
    return DistVarInfo(; bins = cor_type.bins, normalise = cor_type.normalise)
end
function default_dist(::DistCanonical, ::CovLTD)
    return DistLog()
end
function default_dist(::DistCanonical, ::CovDistance)
    return DistCor()
end
function default_dist(::DistCanonical, cor_type::PortfolioOptimiserCovCor)
    dist_type = DistMLP()
    set_absolute_dist!(dist_type, cor_type)
    return dist_type
end
function default_dist(dist_type::DistDistCanonical, cor_type::PortCovCor)
    return default_dist(dist_type, cor_type.ce)
end
function default_dist(dist::DistDistCanonical, cor_type::CovMutualInfo)
    return DistDistVarInfo(;
                           de = DistVarInfo(; bins = cor_type.bins,
                                            normalise = cor_type.normalise),
                           distance = dist.distance, args = dist.args, kwargs = dist.kwargs)
end
function default_dist(dist::DistDistCanonical, ::CovLTD)
    return DistDistLog(; distance = dist.distance, args = dist.args, kwargs = dist.kwargs)
end
function default_dist(dist::DistDistCanonical, ::CovDistance)
    return DistDistCor(; distance = dist.distance, args = dist.args, kwargs = dist.kwargs)
end
function default_dist(dist::DistDistCanonical, cor_type::PortfolioOptimiserCovCor)
    dist_type = DistDistMLP(; distance = dist.distance, args = dist.args,
                            kwargs = dist.kwargs)
    set_absolute_dist!(dist_type, cor_type)
    return dist_type
end
function default_dist(dist_type::DistType, cor_type::PortfolioOptimiserCovCor)
    set_absolute_dist!(dist_type, cor_type)
    return dist_type
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
    xjl, xju = extrema(xj)
    k1 = (xju - xjl) / pyconvert(eltype(xj), bin_width_func(Py(xj).to_numpy()))
    return round(Int,
                 if j != i
                     xil, xiu = extrema(xi)
                     k2 = (xiu - xil) /
                          pyconvert(eltype(xi), bin_width_func(Py(xi).to_numpy()))
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
function variation_info(X::AbstractMatrix, bins::Union{<:AbstractBins, <:Integer} = HGR(),
                        normalise::Bool = true)
    T, N = size(X)
    var_mtx = Matrix{eltype(X)}(undef, N, N)

    bin_width_func = _bin_width_func(bins)

    for j ∈ axes(X, 2)
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
#=
function mutual_variation_info(X::AbstractMatrix,
                               bins::Union{<:AbstractBins, <:Integer} = Knuth(),
                               normalise::Bool = true)
    T, N = size(X)
    mut_mtx = Matrix{eltype(X)}(undef, N, N)
    var_mtx = Matrix{eltype(X)}(undef, N, N)

    bin_width_func = _bin_width_func(bins)

    for j ∈ axes(X, 2)
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

export dist, calc_num_bins, calc_hist_data, variation_info
