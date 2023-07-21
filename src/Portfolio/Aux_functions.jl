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

function scokurt(returns, args...; minval = 0.0, mean_func::Function = mean, kwargs...)
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

export cokurt, scokurt