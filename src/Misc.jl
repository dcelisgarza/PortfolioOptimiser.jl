_nanmean(x) = mean(filter(!isnan, x))
nanmean(x; dims = 1) = ndims(x) > 1 ? mapslices(_nanmean, x, dims = dims) : _nanmean(x)

function sma(a::Array, n::Integer)
    vals = zeros(size(a, 1) - (n - 1), size(a, 2))

    for j in 1:size(a, 2)
        for i in 1:(size(a, 1) - (n - 1))
            vals[i, j] = nanmean(a[i:(i + (n - 1)), j])
        end
    end

    vals
end

function ema(a::Array, n::Integer; wilder = false)
    k = (1 + !wilder) / (n + !wilder)

    vals = zeros(size(a, 1), size(a, 2))
    # seed with first value with an sma value
    vals[n, :] = sma(a, n)[1, :]

    for j in 1:size(a, 2)
        for i in (n + 1):size(a, 1)
            vals[i, j] = a[i, j] * k + vals[i - 1, j] * (1 - k)
        end
    end

    vals[n:end, :]
end