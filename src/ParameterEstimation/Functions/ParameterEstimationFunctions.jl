
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

include("./PosdefFixFunctions.jl")
include("./MatrixDenoisingFunctions.jl")
include("./DistanceMatrixFunctions.jl")
include("./ClusteringFunctions.jl")
include("./CovCorKurtSkewFunctions.jl")
include("./MeanEstimatorFunctions.jl")
include("./WorstCaseFunctions.jl")
include("./RegressionFunctions.jl")
include("./BlackLittermanFunctions.jl")
# include("./NetworkFunctions.jl")
include("./OWAFunctions.jl")

export cov_returns
