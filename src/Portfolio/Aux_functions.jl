function block_vec_pq(A, p, q)
    mp, nq = size(A)

    !(mod(mp, p) == 0 && mod(nq, p) == 0) && (throw(
        DimensionMismatch(
            "dimensions A, $(size(A)), must be integer multiples of (p, q) = ($p, $q)",
        ),
    ))

    m = Int(mp / p)
    n = Int(nq / q)

    A_vec = Matrix{eltype(A)}(undef, m * n, p * q)
    for j in 0:(n - 1)
        Aj = Matrix{eltype(A)}(undef, m, p * q)
        for i in 0:(m - 1)
            Aij = vec(A[(1 + (i * p)):((i + 1) * p), (1 + (j * q)):((j + 1) * q)])
            Aj[i + 1, :] .= Aij
        end
        A_vec[(1 + (j * m)):((j + 1) * m), :] .= Aj
    end

    return A_vec
end

function cokurt(x::AbstractMatrix, mean_func::Function = mean, args...; kwargs...)
    T, N = size(x)
    mu =
        !haskey(kwargs, :dims) ? mean_func(x, args...; dims = 1, kwargs...) :
        mean_func(x, args...; kwargs...)
    y = x .- mu
    ex = eltype(y)
    o = transpose(range(start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    cokurt = transpose(z) * z / T
    return cokurt
end

function scokurt(
    x::AbstractMatrix,
    minval = 0.0,
    mean_func::Function = mean,
    args...;
    kwargs...,
)
    T, N = size(x)
    mu =
        !haskey(kwargs, :dims) ? mean_func(x, args...; dims = 1, kwargs...) :
        mean_func(x, args...; kwargs...)
    y = x .- mu
    y .= min.(y, minval)
    ex = eltype(y)
    o = transpose(range(start = one(ex), stop = one(ex), length = N))
    z = kron(o, y) .* kron(y, o)
    scokurt = transpose(z) * z / T
    return scokurt
end

function cokurt(returns, mean_func::Function = mean, args...; kwargs...)
    nms = names(returns)
    cols = vec(["$x-$y" for x in nms, y in nms])
    x = Matrix(returns)
    cokrt = cokurt(x, mean_func, args...; kwargs...)
    df = DataFrame(cokrt, cols)
    return df
end

function scokurt(returns, minval = 0.0, mean_func::Function = mean, args...; kwargs...)
    nms = names(returns)
    cols = vec(["$x-$y" for x in nms, y in nms])
    x = Matrix(returns)
    scokrt = scokurt(x, minval, mean_func, args...; kwargs...)
    df = DataFrame(scokrt, cols)
    return df
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

function cov_returns(x; seed = nothing, rng = Random.default_rng(), len = 10, iters = 5)
    !isnothing(seed) && Random.seed!(rng, seed)

    n = size(x)[1]
    a = randn(rng, n + len, n)

    for _ in 1:iters
        _cov = cov(a)
        _C = cholesky(_cov)
        a .= transpose(_C.L \ transpose(a))
        _cov = cov(a)
        _desv = transpose(sqrt.(diag(_cov)))
        a .= (a .- mean(a, dims = 1)) ./ _desv
    end

    C = cholesky(x)
    return a * C.U
end

cov_returns(cov)

covar = transpose(
    reshape(
        [
            1.89630827,
            2.1673679,
            1.58698645,
            1.60729798,
            1.96541505,
            1.81717423,
            2.01480903,
            1.73237143,
            1.40397697,
            1.86803613,
            2.1673679,
            4.5471796,
            3.68287909,
            2.22016266,
            2.98132602,
            3.7163201,
            2.67612745,
            3.1728205,
            3.278824,
            3.68953981,
            1.58698645,
            3.68287909,
            3.49336691,
            1.68310924,
            2.30222573,
            2.97821139,
            1.72657227,
            2.37405877,
            2.93161501,
            3.07048313,
            1.60729798,
            2.22016266,
            1.68310924,
            2.75212897,
            1.63011175,
            1.82311827,
            2.22875933,
            1.96265296,
            2.08125574,
            2.3565411,
            1.96541505,
            2.98132602,
            2.30222573,
            1.63011175,
            2.74932335,
            2.593768,
            2.40633646,
            2.44451217,
            2.13873155,
            2.34363631,
            1.81717423,
            3.7163201,
            2.97821139,
            1.82311827,
            2.593768,
            3.35546245,
            2.45031909,
            2.72600372,
            2.64109986,
            3.1736573,
            2.01480903,
            2.67612745,
            1.72657227,
            2.22875933,
            2.40633646,
            2.45031909,
            3.19101728,
            3.01317032,
            2.34484918,
            2.52663068,
            1.73237143,
            3.1728205,
            2.37405877,
            1.96265296,
            2.44451217,
            2.72600372,
            3.01317032,
            3.74601598,
            2.92414134,
            3.04099192,
            1.40397697,
            3.278824,
            2.93161501,
            2.08125574,
            2.13873155,
            2.64109986,
            2.34484918,
            2.92414134,
            4.20246349,
            3.14882805,
            1.86803613,
            3.68953981,
            3.07048313,
            2.3565411,
            2.34363631,
            3.1736573,
            2.52663068,
            3.04099192,
            3.14882805,
            3.63686227,
        ],
        10,
        10,
    ),
)