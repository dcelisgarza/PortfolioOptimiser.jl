"""
"""
function risk_matrix(
    type::AbstractRiskModel,
    returns;
    target = 1.02^(1 / 252) - 1, # Daily risk free rate.
    fix_method::Union{SFix, DFix} = SFix(),
    freq = 252,
    span = Int(ceil(freq / 1.4)),
)
    if typeof(type) <: Cov
        return risk_matrix(Cov(), returns; fix_method = fix_method, freq = freq)
    elseif typeof(type) <: SCov
        return risk_matrix(
            SCov(),
            returns;
            target = target,
            fix_method = fix_method,
            freq = freq,
        )
    elseif typeof(type) <: ECov
        return risk_matrix(
            ECov(),
            returns;
            fix_method = fix_method,
            freq = freq,
            span = span,
        )
    elseif typeof(type) <: ESCov
        return risk_matrix(
            ESCov(),
            returns;
            target = target,
            fix_method = fix_method,
            freq = freq,
            span = span,
        )
    end
end

function risk_matrix(::Cov, returns; fix_method::Union{SFix, DFix} = SFix(), freq = 252)
    return make_pos_def(fix_method, cov(returns) * freq)
end

function risk_matrix(
    ::SCov,
    returns;
    target = 1.02^(1 / 252) - 1, # Daily risk free rate.
    fix_method::Union{SFix, DFix} = SFix(),
    freq = 252,
)
    semi_ret = min.(returns .- target, 0)

    return make_pos_def(fix_method, cov(SimpleCovariance(), semi_ret; mean = 0) * freq)
end

function risk_matrix(
    ::ECov,
    returns;
    fix_method::Union{SFix, DFix} = SFix(),
    freq = 252,
    span = Int(ceil(freq / 1.4)),
)
    N = size(returns, 1)

    make_pos_def(fix_method, cov(returns, eweights(N, 2 / (span + 1))) * freq)
end

function risk_matrix(
    ::ESCov,
    returns;
    target = 1.02^(1 / 252) - 1, # Daily risk free rate.
    fix_method::Union{SFix, DFix} = SFix(),
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
