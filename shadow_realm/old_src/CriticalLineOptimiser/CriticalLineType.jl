abstract type AbstractCriticalLine <: AbstractPortfolioOptimiser end

struct CriticalLine{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11} <: AbstractCriticalLine
    tickers::T1
    mean_ret::T2
    weights::T3
    cov_mtx::T4
    lower_bounds::T5
    upper_bounds::T6
    w::T7
    lambda::T8
    gamma::T9
    free::T10
    frontier_values::T11
end
function CriticalLine(tickers, mean_ret, cov_mtx; weight_bounds = (0, 1))
    num_tickers = length(tickers)

    @assert num_tickers == length(mean_ret) == size(cov_mtx, 1) == size(cov_mtx, 2)

    lower_bounds, upper_bounds = _create_weight_bounds(num_tickers, weight_bounds)

    CriticalLine(
        tickers,
        mean_ret,
        zeros(num_tickers),
        cov_mtx,
        lower_bounds,
        upper_bounds,
        Vector{Vector{Float64}}(),
        Vector{Union{Float64, Nothing}}(),
        Vector{Union{Float64, Nothing}}(),
        Vector{Vector{Float64}}(),
        Vector{
            NamedTuple{
                (:mu, :sigma, :weights),
                Tuple{Vector{Float64}, Vector{Float64}, Vector{Vector{Float64}}},
            },
        }(
            undef,
            1,
        ),
    )
end
