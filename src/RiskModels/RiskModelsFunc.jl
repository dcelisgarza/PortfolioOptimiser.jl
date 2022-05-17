"""
Calculate covariance matrices.

## Wrapper function

```
risk_matrix(
    type::AbstractRiskModel,
    returns;
    target = 1.02^(1 / 252) - 1,
    fix_method::Union{SFix, DFix} = SFix(),
    freq = 252,
    span = Int(ceil(freq / 1.4)),
)
```

Dispatches specific covariance matrix calculation methods according to `type.`

- `returns`: array of asset returns, each column is an asset, each row an entry.
- `target`: target for distinguishing "upside" and "downside" returns. Defaults to the daily risk-free rate.
- `fix_method`: the fixing method used in case the matrix is non-positive definite.
- `freq`: frequency at which returns are recorded, defaults to the average number of trading days in a year.
- `span`: span parmeter for calculating the exponential weights, `α = 2 / (span + 1)`.

## Normal covariance

```
risk_matrix(::Cov, returns; fix_method::Union{SFix, DFix} = SFix(), freq = 252)
```

## Semi-covariance

The semi-covariance sets a target for distinguishing "upside" and "downside" risk. It only considers fluctuations below `target` which heavily biases losses. Lets users minimise negative risk.

```
risk_matrix(
    ::SCov,
    returns;
    target = 1.02^(1 / 252) - 1,
    fix_method::Union{SFix, DFix} = SFix(),
    freq = 252,
)
```

## Exponentially weighted covariance

The exponentially weighted covariance uses exponentially weighted expected returns rather than normally weighted expected returns. More recent returns are weighted more heavily.

```
risk_matrix(
    ::ECov,
    returns;
    fix_method::Union{SFix, DFix} = SFix(),
    freq = 252,
    span = Int(ceil(freq / 1.4)),
)
```

## Exponentially weighted semi-covariance

Combines the semi-covariance and exponentially weighted covariance.

```
risk_matrix(
    ::ESCov,
    returns;
    target = 1.02^(1 / 252) - 1, # Daily risk free rate.
    fix_method::Union{SFix, DFix} = SFix(),
    freq = 252,
    span = Int(ceil(freq / 1.4)),
)
```
"""
function risk_matrix(
    type::AbstractRiskModel,
    returns;
    target = 1.02^(1 / 252) - 1,
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
