function risk_matrix(
    ::SampleCov,
    returns;
    fix_method::Union{SpecFix, DiagFix} = SpecFix(),
    freq = 252,
)
    return make_pos_def(fix_method, cov(returns) * freq)
end

function risk_matrix(
    ::SemiCov,
    returns;
    target = 1.02^(1 / 252) - 1, # Daily risk free rate.
    fix_method::Union{SpecFix, DiagFix} = SpecFix(),
    freq = 252,
)
    semi_ret = min.(returns .- target, 0)

    return make_pos_def(fix_method, cov(SimpleCovariance(), semi_ret; mean = 0) * freq)
end

function risk_matrix(
    ::ExpCov,
    returns;
    fix_method::Union{SpecFix, DiagFix} = SpecFix(),
    freq = 252,
    span = Int(ceil(freq / 1.4)),
)
    N = size(returns, 1)

    make_pos_def(fix_method, cov(returns, eweights(N, 2 / (span + 1))) * freq)
end

function risk_matrix(
    ::ExpSemiCov,
    returns;
    target = 1.02^(1 / 252) - 1, # Daily risk free rate.
    fix_method::Union{SpecFix, DiagFix} = SpecFix(),
    freq = 252,
    span = Int(ceil(freq / 1.4)),
)
    N = size(returns, 1)

    semi_ret = min.(returns .- target, 0)

    return make_pos_def(
        fix_method,
        cov(SimpleCovariance(), semi_ret, eweights(N, 2 / (span + 1)); mean = 0) * freq,
    )
end
