
function cov_returns(x; iters = 5, len = 10, rng = Random.default_rng(), seed = nothing)
    !isnothing(seed) && Random.seed!(rng, seed)

    n = size(x, 1)
    a = randn(rng, n + len, n)

    for _ in 1:iters
        _cov = cov(a)
        _C = cholesky(_cov)
        a .= a * (_C.U \ I)
        _cov = cov(a)
        _s = transpose(sqrt.(diag(_cov)))
        a .= (a .- mean(a, dims = 1)) ./ _s
    end

    C = cholesky(x)
    return a * C.U
end

export cov_returns
