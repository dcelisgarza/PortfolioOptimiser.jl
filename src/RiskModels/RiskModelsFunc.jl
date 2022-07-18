import Statistics.cov
"""
Calculate covariance matrices.

## Wrapper function

```
cov(
    type::AbstractRiskModel,
    returns;
    target = 1.02^(1 / 252) - 1,
    fix_method::AbstractFixPosDef = SFix(),
    freq = 1,
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
cov(::Cov, returns; fix_method::AbstractFixPosDef = SFix(), freq = 1)
```

## Semi-covariance

The semi-covariance sets a target for distinguishing "upside" and "downside" risk. It only considers fluctuations below `target` which heavily biases losses. Lets users minimise negative risk.

```
cov(
    ::SCov,
    returns;
    target = 1.02^(1 / 252) - 1,
    fix_method::AbstractFixPosDef = SFix(),
    freq = 1,
)
```

## Exponentially weighted covariance

The exponentially weighted covariance uses exponentially weighted expected returns rather than normally weighted expected returns. More recent returns are weighted more heavily.

```
cov(
    ::ECov,
    returns;
    fix_method::AbstractFixPosDef = SFix(),
    freq = 1,
    span = Int(ceil(freq / 1.4)),
)
```

## Exponentially weighted semi-covariance

Combines the semi-covariance and exponentially weighted covariance.

```
cov(
    ::ESCov,
    returns;
    target = 1.02^(1 / 252) - 1, # Daily risk free rate.
    fix_method::AbstractFixPosDef = SFix(),
    freq = 1,
    span = Int(ceil(freq / 1.4)),
)
```
"""
function cov(
    type::AbstractRiskModel,
    returns,
    target = 1.02^(1 / 252) - 1,
    fix_method::AbstractFixPosDef = SFix(),
    freq = 1,
    span = Int(
        ceil(
            size(returns, 1) * log2(min(size(returns, 1), freq)) /
            log2(max(size(returns, 1), freq)),
        ),
    ),
    scale = nothing,
    custom_cov_estimator = nothing,
    custom_cov_args = (),
    custom_cov_kwargs = (),
)
    if typeof(type) <: Cov
        return cov(Cov(), returns; fix_method = fix_method, freq = freq, scale = scale)
    elseif typeof(type) <: SCov
        return cov(
            SCov(),
            returns;
            target = target,
            fix_method = fix_method,
            freq = freq,
            scale = scale,
        )
    elseif typeof(type) <: ECov
        return cov(
            ECov(),
            returns;
            fix_method = fix_method,
            freq = freq,
            span = span,
            scale = scale,
        )
    elseif typeof(type) <: ESCov
        return cov(
            ESCov(),
            returns;
            target = target,
            fix_method = fix_method,
            freq = freq,
            span = span,
            scale = scale,
        )
    elseif typeof(type) <: CustomCov
        cov(
            CustomCov(),
            returns;
            freq = freq,
            estimator = custom_cov_estimator,
            args = custom_cov_args,
            kwargs = custom_cov_kwargs,
        )
    elseif typeof(type) <: CustomSCov
        cov(
            CustomSCov(),
            returns;
            target = target,
            freq = freq,
            estimator = custom_cov_estimator,
            args = custom_cov_args,
            kwargs = custom_cov_kwargs,
        )
    end
end

function cov(
    ::Cov,
    returns;
    fix_method::AbstractFixPosDef = SFix(),
    freq = 1,
    scale = nothing,
)
    return make_pos_def(fix_method, cov(returns) * freq, scale)
end

function cov(
    ::SCov,
    returns;
    target = 1.02^(1 / 252) - 1, # Daily risk free rate.
    fix_method::AbstractFixPosDef = SFix(),
    freq = 1,
    scale = nothing,
)
    semi_returns = min.(returns .- target, 0)

    return make_pos_def(
        fix_method,
        cov(SimpleCovariance(), semi_returns; mean = 0) * freq,
        scale,
    )
end

function cov(
    ::ECov,
    returns;
    fix_method::AbstractFixPosDef = SFix(),
    freq = 1,
    span = Int(
        ceil(
            size(returns, 1) * log2(min(size(returns, 1), freq)) /
            log2(max(size(returns, 1), freq)),
        ),
    ),
    scale = nothing,
)
    N = size(returns, 1)

    make_pos_def(fix_method, cov(returns, eweights(N, 2 / (span + 1))) * freq, scale)
end

function cov(
    ::ESCov,
    returns;
    target = 1.02^(1 / 252) - 1, # Daily risk free rate.
    fix_method::AbstractFixPosDef = SFix(),
    freq = 1,
    span = Int(
        ceil(
            size(returns, 1) * log2(min(size(returns, 1), freq)) /
            log2(max(size(returns, 1), freq)),
        ),
    ),
    scale = nothing,
)
    N = size(returns, 1)

    semi_returns = min.(returns .- target, 0)

    return make_pos_def(
        fix_method,
        cov(SimpleCovariance(), semi_returns, eweights(N, 2 / (span + 1)); mean = 0) * freq,
        scale,
    )
end

function cov(::CustomCov, returns; freq = 1, estimator = nothing, args = (), kwargs = ())
    return isnothing(estimator) ? cov(returns, args...; kwargs...) * freq :
           cov(estimator, returns, args...; kwargs...) * freq
end

function cov(
    ::CustomSCov,
    returns;
    target = 1.02^(1 / 252) - 1,
    freq = 1,
    estimator = nothing,
    args = (),
    kwargs = (),
)
    semi_returns = min.(returns .- target, 0)

    return isnothing(estimator) ? cov(semi_returns, args...; mean = 0, kwargs...) * freq :
           cov(estimator, semi_returns, args...; mean = 0, kwargs...) * freq
end