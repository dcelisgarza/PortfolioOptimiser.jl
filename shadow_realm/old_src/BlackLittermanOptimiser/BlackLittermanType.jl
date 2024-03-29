abstract type AbstractBlackLitterman <: AbstractPortfolioOptimiser end

struct BlackLitterman{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14} <:
       AbstractBlackLitterman
    rf::T1
    risk_aversion::T2
    tau::T3
    tickers::T4
    weights::T5
    cov_mtx::T6
    Q::T7
    P::T8
    pi::T9
    omega::T10
    tau_sigma_p::T11
    A::T12
    post_ret::T13
    post_cov::T14
end

function BlackLitterman(tickers::AbstractArray, cov_mtx::AbstractArray;
                        rf::Real = 1.02^(1 / 252) - 1, risk_aversion::Real = 1,
                        tau::Real = 0.05, omega::Union{AbstractArray, Symbol} = :default, # either a square matrix, :idzorek, :default
                        pi::Union{Nothing, AbstractArray, Symbol} = nothing, # either a vector, `nothing`, `:Equal`, or `:market`
                        absolute_views::Union{Nothing, Dict} = nothing,
                        Q::Union{Nothing, AbstractArray} = nothing,
                        P::Union{Nothing, AbstractArray} = nothing,
                        view_confidence::Union{Nothing, AbstractArray} = nothing,
                        market_caps::Union{Nothing, AbstractArray} = nothing,)
    num_tickers = length(tickers)
    @assert size(cov_mtx) == (num_tickers, num_tickers)

    if isnothing(absolute_views)
        @assert !isnothing(Q) "if not providing an absolute_views dictionary, must provide a Q vector"
        if isnothing(P) && length(Q) == num_tickers
            # If there's no P and views are provided on every asset, P defaults to the identity matrix.
            # P = I[num_tickers]
            P = Diagonal(I, num_tickers)
        else
            @assert !isnothing(P) "if Q does not have an entry for every ticker, must provide a P matrix"
        end
    else
        Q, P = _parse_views(tickers, absolute_views)
    end
    K = length(Q)
    @assert size(P) == (num_tickers, K) "size of P must be equal to ($num_tickers, $K), it is equal to $(size(P))"

    _val_compare_benchmark(risk_aversion, <=, 0, 1, "risk_aversion")

    if typeof(pi) <: AbstractVector
        @assert length(pi) == num_tickers
    elseif isnothing(pi)
        @warn("running Black-Litterman with no prior")
        pi = zeros(num_tickers)
    elseif pi == :market
        @assert !isnothing(market_caps) "please provide a vector of market caps via the market_caps keyword"
        pi = market_implied_prior_returns(market_caps, cov_mtx, risk_aversion, rf)
    elseif pi == :Equal
        pi = ones(num_tickers) / num_tickers
    else
        throw(ArgumentError("pi must be either nothing, :market or :Equal"))
    end

    _val_compare_benchmark(tau, <=, 0, 0.05, "tau")
    _val_compare_benchmark(tau, >, 1, 0.05, "tau")

    if typeof(omega) <: AbstractArray
        # Do nothing, all is good.
    elseif omega == :idzorek
        @assert !isnothing(view_confidence) "to use Idzorek's method, `view_confidence` needs to be a vector of percentage confidence levels for each view"

        @assert length(view_confidence) == K "the length of `view_confidence` must be equal to the number of views privided in `Q`: $K"

        omega = idzorek(view_confidence, cov_mtx, Q, P, tau)
    elseif omega == :default
        omega = Diagonal(tau * P' * cov_mtx * P)
    else
        throw(ArgumentError("omega: $omega must be a square matrix, `:idzorek`, `:default`, or a $(K)x$(K) matrix"))
    end
    @assert size(omega) == (K, K)

    # Intermediate values
    tau_sigma_p = tau * cov_mtx * P
    # NxN * NxK => NxK
    A = P' * tau_sigma_p + omega
    # KxN * NxK + KxK => KxK + KxK => KxK

    # Posterior return
    b = Q - P' * pi
    # Kx1 - KxN * Nx1 => Kx1 - Kx1 => Kx1
    post_ret = pi + tau_sigma_p * (A \ b)
    # Nx1 + NxK * (KxK \ Kx1) => Nx1 + NxK * Kx1 => Nx1 + Nx1 => Nx1

    # Posterior covariance
    M = tau * cov_mtx - tau_sigma_p * (A \ tau_sigma_p')
    # NxN - NxK * (KxK \ KxN) => NxN - NxK * KxN => NxN - NxN => NxN
    post_cov = cov_mtx + M

    # Weights
    weights = (risk_aversion * cov_mtx) \ post_ret
    weights /= sum(weights)

    return BlackLitterman(rf, risk_aversion, tau, tickers, weights, cov_mtx, Q, P, pi,
                          omega, tau_sigma_p, A, post_ret, post_cov)
end
