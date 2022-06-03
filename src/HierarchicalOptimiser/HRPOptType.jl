abstract type AbstractHRPOpt <: AbstractPortfolioOptimiser end

struct HRPOpt{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <: AbstractHRPOpt
    tickers::T1
    mean_ret::T2
    weights::T3
    returns::T4
    cov_mtx::T5
    rf::T6
    freq::T7
    risk_aversion::T8
    linkage::T9
    clusters::T10
end
function HRPOpt(
    tickers::AbstractVector{<:AbstractString};
    linkage::Symbol = :single,
    returns = nothing,
    cov_mtx = nothing,
    mean_ret = nothing,
    rf = 0.02,
    freq = 252,
    risk_aversion = 1,
    D = :default,
)
    if isnothing(returns) && isnothing(cov_mtx)
        throw(ArgumentError("Either returns or cov_mtx must be defined."))
    elseif isnothing(returns)
        @assert size(cov_mtx, 1) == size(cov_mtx, 2) == length(tickers)
        cor_mtx = cov2cor(cov_mtx)
    elseif isnothing(cov_mtx)
        @assert size(returns, 2) == length(tickers)
        cov_mtx = cov(returns)
        cor_mtx = cor(returns)
    else
        @assert size(cov_mtx, 1) == size(cov_mtx, 2) == size(returns, 2) == length(tickers)
        cor_mtx = cov2cor(cov_mtx)
    end

    if D == :default
        D = Symmetric(sqrt.(clamp.((1 .- cor_mtx) / 2, 0, 1)))
    elseif typeof(D) <: AbstractArray
        @assert size(D) == size(cov_mtx)
    else
        throw(
            ArgumentError(
                "Distance matrix D must be :default, or a square matrix if size equal to the covariance matrix: $(size(cov_mtx)).",
            ),
        )
    end
    clusters = hclust(D, linkage = linkage)

    weights = zeros(length(tickers))

    risk_aversion = _val_compare_benchmark(risk_aversion, <=, 0, 1, "risk_aversion")

    return HRPOpt(
        tickers,
        mean_ret,
        weights,
        returns,
        cov_mtx,
        rf,
        freq,
        risk_aversion,
        linkage,
        clusters,
    )
end
