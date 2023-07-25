function asset_statistics!(
    portfolio::Portfolio,
    target_ret::AbstractFloat = 0.0,
    mean_func::Function = mean,
    cov_func::Function = cov,
    mean_args = (),
    cov_args = ();
    mean_kwargs = (;),
    cov_kwargs = (;),
)
    N = ncol(portfolio.returns) - 1

    mu = vec(
        !haskey(mean_kwargs, :dims) ?
        mean_func(
            Matrix(portfolio.returns[!, 2:end]),
            mean_args...;
            dims = 1,
            mean_kwargs...,
        ) :
        mean_func(Matrix(portfolio.returns[!, 2:end]), mean_args...; mean_kwargs...),
    )

    portfolio.mu = DataFrame(tickers = names(portfolio.returns)[2:end], val = mu)
    cov_mtx = cov_func(Matrix(portfolio.returns[!, 2:end]), cov_args...; cov_kwargs...)
    nms = names(portfolio.returns[!, 2:end])
    portfolio.cov = DataFrame(cov_mtx, nms)
    portfolio.kurt = cokurt(portfolio.returns[!, 2:end], transpose(mu))
    portfolio.skurt = scokurt(portfolio.returns[!, 2:end], transpose(mu), target_ret)
    missing, portfolio.L_2, portfolio.S_2 = dup_elim_sum_matrices(N)

    return nothing
end

export asset_statistics!