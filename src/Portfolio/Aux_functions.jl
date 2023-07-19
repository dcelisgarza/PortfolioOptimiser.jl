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

function cokurt(returns)
    nms = names(returns)[2:end]
    cols = vec(["$x-$y" for x in nms, y in nms])
    x = Matrix(returns[!, 2:end])
    T, N = size(x)
    mu = mean(x, dims = 1)
    x .-= mu
    o = ones(1, N)
    z = kron(o, x) .* kron(x, o)
    cokurt = transpose(z) * z / T

    df = DataFrame(cokurt, cols)
    return df
end

function scokurt(returns, minval = 0)
    nms = names(returns)[2:end]
    cols = vec(["$x-$y" for x in nms, y in nms])
    x = Matrix(returns[!, 2:end])
    T, N = size(x)
    mu = mean(x, dims = 1)
    x .-= mu
    x .= min.(x, minval)
    o = ones(1, N)
    z = kron(o, x) .* kron(x, o)
    scokurt = transpose(z) * z / T

    df = DataFrame(scokurt, cols)
    return df
end