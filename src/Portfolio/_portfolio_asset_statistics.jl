function asset_statistics!(portfolio::Portfolio)
    returns = Matrix(portfolio.returns[!, 2:end])
    T, N = size(returns)

    missing, portfolio.L_2, portfolio.S_2 = dup_elim_sum_matrices(N)
end

export asset_statistics!