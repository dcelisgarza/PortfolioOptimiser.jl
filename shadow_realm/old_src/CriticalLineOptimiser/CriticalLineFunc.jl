function min_risk!(portfolio::AbstractCriticalLine)
    w = portfolio.w
    if isempty(w)
        _solve!(portfolio)
    end
    var = Float64[]
    cov_mtx = portfolio.cov_mtx
    for wi in w
        a = dot(wi, cov_mtx, wi)
        push!(var, a)
    end
    portfolio.weights .= w[argmin(var)]
    return nothing
end

function max_sharpe!(portfolio::AbstractCriticalLine)
    w = portfolio.w
    if isempty(w)
        _solve!(portfolio)
    end
    # 1) Compute local max SR portfolio between two neighboring turning points
    w_sr = Vector{Vector{Float64}}()
    sr = Float64[]
    for i in 1:(length(w) - 1)
        w0 = copy(w[i])
        w1 = copy(w[i + 1])
        args = (w0, w1)
        a, b = _golden_section(portfolio, _eval_sr, 0, 1; args = args)
        push!(w_sr, a * w0 + (1 - a) * w1)
        push!(sr, b)
    end
    portfolio.weights .= w_sr[argmax(sr)]
    return nothing
end

function efficient_frontier!(portfolio::AbstractCriticalLine, points = 100)
    if isempty(portfolio.w)
        _solve!(portfolio)
    end

    mean_ret = portfolio.mean_ret
    cov_mtx = portfolio.cov_mtx

    lw = length(portfolio.w)
    mu, sigma, weights = Float64[], Float64[], Vector{Float64}[]
    # remove the 1, to avoid duplications
    A = range(0, 1; length = div(points, lw))
    a = A[1:(end - 1)]
    b = 1:(lw - 1)
    for i in b
        w0 = portfolio.w[i]
        w1 = portfolio.w[i + 1]
        if i == b[end]
            # include the 1 in the last iteration
            a = A
        end
        for j in a
            w = w1 * j + (1 - j) * w0
            push!(weights, copy(w))
            push!(mu, port_return(w, mean_ret))
            push!(sigma, sqrt(port_variance(w, cov_mtx)))
        end
    end
    portfolio.frontier_values[1] = (mu = mu, sigma = sigma, weights = weights)
    return nothing
end

@inline function _eval_sr(portfolio::AbstractCriticalLine, a, w0, w1)
    # Evaluate SR of the portfolio within the convex combination
    mean_ret = portfolio.mean_ret
    cov_mtx = portfolio.cov_mtx
    w = a * w0 + (1 - a) * w1
    mu = port_return(w, mean_ret)
    sigma = sqrt(port_variance(w, cov_mtx))
    return mu / sigma
end

function _golden_section(portfolio::AbstractCriticalLine, obj::Function, a, b;
                         args = nothing, minimum = false, tol = 1e-9,)
    # Golden section method. Maximum if minimum == false is passed
    minimum ? sign = 1 : sign = -1
    max_iter = ceil(Int, -2.078087 * log(tol / abs(b - a)))

    r = 0.618033989
    c = 1.0 - r
    # Initialize
    x1 = r * a + c * b
    x2 = c * a + r * b
    f1 = sign * obj(portfolio, x1, args...)
    f2 = sign * obj(portfolio, x2, args...)
    # Loop
    for _ in 1:max_iter
        if f1 > f2
            a = x1
            x1 = x2
            f1 = f2
            x2 = c * a + r * b
            f2 = sign * obj(portfolio, x2, args...)
        else
            b = x2
            x2 = x1
            f2 = f1
            x1 = r * a + c * b
            f1 = sign * obj(portfolio, x1, args...)
        end
    end
    if f1 < f2
        return x1, sign * f1
    else
        return x2, sign * f2
    end
end

function _solve!(portfolio::AbstractCriticalLine)
    f, w = _init(portfolio)

    push!(portfolio.w, copy(w))
    push!(portfolio.lambda, nothing)
    push!(portfolio.gamma, nothing)
    push!(portfolio.free, f)

    lower_bounds = portfolio.lower_bounds
    upper_bounds = portfolio.upper_bounds

    mean_ret = portfolio.mean_ret
    num_tickers = length(mean_ret)

    i_in = nothing
    bi_in = nothing
    i_out = nothing
    while true
        # 1) case a): bound one free weight_bounds
        l_in = nothing
        if length(f) > 1
            cov_f, cov_fb, mean_f, w_b = _get_matrices(portfolio, f)
            icov_f = inv(cov_f)
            j = 1
            for i in f
                l, bi = _compute_lambda(icov_f, cov_fb, mean_f, w_b, j,
                                        (lower_bounds[i], upper_bounds[i]))
                if _infnone(l) > _infnone(l_in)
                    l_in, i_in, bi_in = l, i, bi
                end
                j += 1
            end
        end

        # 2) case b): Free one bounded weight
        l_out = nothing
        if length(f) < length(mean_ret)
            b = setdiff(1:num_tickers, f)
            for i in b
                cov_f, cov_fb, mean_f, w_b = _get_matrices(portfolio, [f; i])
                icov_f = inv(cov_f)
                l, bi = _compute_lambda(icov_f, cov_fb, mean_f, w_b, length(mean_f),
                                        portfolio.w[end][i])

                if (isnothing(portfolio.lambda[end]) || l < portfolio.lambda[end]) &&
                   l > _infnone(l_out)
                    l_out, i_out = l, i
                end
            end
        end

        # 3) compute minimum variance solution
        if (isnothing(l_in) || l_in < 0) && (isnothing(l_out) || l_out < 0)
            push!(portfolio.lambda, 0)
            cov_f, cov_fb, mean_f, w_b = _get_matrices(portfolio, f)
            icov_f = inv(cov_f)
            mean_f .= 0
        else
            # 4) decide lambda
            if _infnone(l_in) > _infnone(l_out)
                push!(portfolio.lambda, l_in)
                popat!(f, findfirst(x -> x == i_in, f))
                w[i_in] = bi_in # set value at the correct boundary
            else
                push!(portfolio.lambda, l_out)
                push!(f, i_out)
            end
            cov_f, cov_fb, mean_f, w_b = _get_matrices(portfolio, f)
            icov_f = inv(cov_f)
        end

        # 5) compute solution vector
        w_f, g = _compute_w(portfolio, icov_f, cov_fb, mean_f, w_b)
        for i in 1:length(f)
            w[f[i]] = w_f[i]
        end
        push!(portfolio.w, copy(w))
        push!(portfolio.gamma, g)
        push!(portfolio.free, f)

        if portfolio.lambda[end] == 0
            break
        end
    end

    _purge_num_err(portfolio, 1e-9)
    return _purge_excess(portfolio)
end

@inline function _init(portfolio::AbstractCriticalLine)
    mean_ret = portfolio.mean_ret
    upper_bounds = portfolio.upper_bounds

    # Get the indices that sort the mean returns in ascending order.
    b = sortperm(mean_ret)

    # Get the number of tickers to start from the last entry, b[length(mean_ret)] is the index of the ticker with the highest mean return.
    i = length(mean_ret) + 1
    w = copy(portfolio.lower_bounds)
    while sum(w) < 1
        i -= 1
        w[b[i]] = upper_bounds[b[i]]
    end
    w[b[i]] += 1 - sum(w)
    return [b[i]], w
end

@inline function _get_matrices(portfolio::AbstractCriticalLine, f)
    mean_ret = portfolio.mean_ret
    cov_mtx = portfolio.cov_mtx
    w = portfolio.w

    cov_f = cov_mtx[f, f]
    mean_f = mean_ret[f]
    b = setdiff(1:length(mean_ret), f)
    cov_fb = cov_mtx[f, b]
    w_b = w[end][b]

    return cov_f, cov_fb, mean_f, w_b
end

@inline function _compute_lambda(icov_f, cov_fb, mean_f, w_b, i, bi)
    # 1) C
    ones_f = ones(length(mean_f))
    c1 = dot(ones_f, icov_f, ones_f)
    c2 = icov_f * mean_f
    c3 = dot(ones_f, icov_f, mean_f)
    c4 = icov_f * ones_f

    c = -c1 * c2[i] + c3 * c4[i]
    if c == 0
        return nothing, nothing
    end

    # 2) bi
    if typeof(bi) <: Tuple
        bi = _compute_bi(c, bi)
    end

    # 3) Lambda
    if isnothing(w_b)
        # All free assets
        return (c4[i] - c1 * bi) / c, bi
    else
        l1 = sum(w_b)
        l2 = icov_f * cov_fb * w_b
        l3 = sum(l2)
        return ((1 - l1 + l3) * c4[i] - c1 * (bi + l2[i])) / c, bi
    end
end

@inline function _compute_bi(c, bi)
    if c > 0
        bi = bi[2]
    else
        bi = bi[1]
    end

    return bi
end

@inline function _compute_w(portfolio, icov_f, cov_fb, mean_f, w_b)
    lambda = portfolio.lambda
    # 1) compute gamma
    ones_f = ones(length(mean_f))
    g1 = dot(ones_f, icov_f, mean_f)
    g2 = dot(ones_f, icov_f, ones_f)
    if isnothing(w_b)
        g, w1 = -lambda[end] * g1 / g2 + 1 / g2, 0
    else
        g3 = sum(w_b)
        g4 = icov_f * cov_fb
        w1 = g4 * w_b
        g4 = sum(w1)
        g = -lambda[end] * g1 / g2 + (1 - g3 + g4) / g2
    end
    # 2) compute weights
    w2 = icov_f * ones_f
    w3 = icov_f * mean_f

    return -w1 + g * w2 + lambda[end] * w3, g
end

@inline function _purge_num_err(portfolio, tol)
    # Remove constraint violations due to ill-conditioned cov matrices.
    lower_bounds = portfolio.lower_bounds
    upper_bounds = portfolio.upper_bounds

    i = 1
    while true
        flag = false
        if i == length(portfolio.w) + 1
            break
        end

        if abs(sum(portfolio.w[i]) - 1) > tol
            flag = true
        else
            for j in 1:length(portfolio.w[i])
                if (portfolio.w[i][j] - lower_bounds[j] < -tol) ||
                   (portfolio.w[i][j] - upper_bounds[j] > tol)
                    flag = true
                    break
                end
            end
        end

        if flag
            popat!(portfolio.w, i)
            popat!(portfolio.lambda, i)
            popat!(portfolio.gamma, i)
            popat!(portfolio.free, i)
        else
            i += 1
        end
    end
end

@inline function _purge_excess(portfolio)
    mean_ret = portfolio.mean_ret
    weights = portfolio.w
    # Remove convex hull violations
    i, repeat = 1, false

    while true
        if !repeat
            (i += 1)
        end
        if i == length(weights)
            break
        end

        w = weights[i]
        mu = port_return(w, mean_ret)
        j, repeat = i + 1, false
        while true
            if j == length(weights) + 1
                break
            end
            w = weights[j]
            mu2 = port_return(w, mean_ret)
            if mu < mu2
                popat!(portfolio.w, i)
                popat!(portfolio.lambda, i)
                popat!(portfolio.gamma, i)
                popat!(portfolio.free, i)
                repeat = true
                break
            else
                j += 1
            end
        end
    end
end

_infnone(x) = isnothing(x) ? -Inf : x
