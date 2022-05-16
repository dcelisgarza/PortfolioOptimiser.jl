"""
```
port_variance(w, cov_mtx)
```

Computes the overall portfolio variance according to the weights `w`, and covariance matrix `cov_mtx`.

The portfolio variance ``\\sigma^2``, is defined as:

```math
\\sigma^2 = \\bm{w}^T\\, \\Sigma\\, \\bm{w}\\,,
```

where ``\\sigma`` is the overall portfolio standard deviation, ``^T`` the transpose operator, ``w`` the weights, and ``\\Sigma`` the covariance matrix.
"""
function port_variance(w, cov_mtx)
    return dot(w, cov_mtx, w)
end

"""
```
port_return(w, mean_ret)
```

Computes the overall portfolio return according to the weights `w`, and mean returns `mean_ret`.

The portfolio return ``\\mu``, is defined as:

```math
\\mu = \\bm{w} \\cdot \\bm{E},
```

where ``\\bm{w}`` are the weights, and ``\\bm{E}`` the mean returns.
"""
function port_return(w, mean_ret)
    return dot(w, mean_ret)
end

"""
```
sharpe_ratio(w, mean_ret, cov_mtx, rf::Real = 0.02)
```

Computes the portfolio's Sharpe ratio given the weights `w`, mean returns `mean_ret`, covariance matrix `cov_mtx`, and risk free rate `rf`.

```
sharpe_ratio(μ, σ, rf::Real = 0.02)
```

Computes the Sharpe ratio given the return `μ`, standard deviation `σ`, and risk free rate `rf`.

The Sharpe ratio ``s``, is defined as:

```math
s = \\dfrac{\\mu - r}{\\sigma},
```

where ``\\mu`` the portfolio's return (see [`port_return`](@ref)), and ``\\sigma`` the portfolio's standard deviation (see [`port_variance`](@ref)). Generally speaking, the greater the Sharpe ratio the better the portfolio. 
!!! note
    The Sharpe ratio penalises large swings in both directions, so assets that tend to have large increases in value are disproportionally penalised by this measure. The Sortino ratio has the same formula but uses an adjusted covariance matrix that accounts only for the negative fluctuations in value. `SemiVar()` and `ExpSemiVar()` in `AbstractRiskModel`](@ref) implement this adjustment, and [`AbstractMeanSemiVar`](@ref) portfolios use it in their optimisations.
"""
function sharpe_ratio(w, mean_ret, cov_mtx, rf::Real = 0.02)
    μ = port_return(w, mean_ret)
    σ = sqrt(port_variance(w, cov_mtx))
    return (μ - rf) / σ
end
function sharpe_ratio(μ, σ, rf::Real = 0.02)
    return (μ - rf) / σ
end

"""
```
L2_reg(w, γ = 1)
```

L2 regularisation. Minimising this reduces the number of negligible weights `w`, with a tuning parameter `γ`. Increasing γ will decrease the number of negligible weights.

The ``L_2`` regularisation is defined as:

```math
L_2 = \\gamma \\Vert \\bm{w} \\Vert\\,,
```

where ``\\gamma`` is the tuning parameter, and ``w`` the weights.
"""
function L2_reg(w, γ = 1)
    return γ * dot(w, w)
end

"""
```
quadratic_utility(w, mean_ret, cov_mtx, risk_aversion = 1)
```

Calculates the quadratic utility given weights `w`, mean returns `mean_ret`, covariance matrix `cov_mtx` and risk aversion `risk_aversion`. Increasing `risk_aversion` decreases risk. 

The quadratic utility ``Q``, is defined as:

```math
Q = \\mu - \\dfrac{1}{2} \\delta  \\bm{w}^T\\, \\Sigma\\, \\bm{w}\\,,
```

where ``\\mu`` is the overall portfolio expected return (see [`port_return`](@ref)), ``\\delta`` the risk aversion, ``w`` the weights, and ``\\Sigma`` the covariance matrix.
"""
function quadratic_utility(w, mean_ret, cov_mtx, risk_aversion = 1)
    μ = port_return(w, mean_ret)
    σ2 = port_variance(w, cov_mtx)
    return μ - 0.5 * risk_aversion * σ2
end

"""
```
semi_ret(returns, benchmark = 0)
```

Compute the historical semi-returns from historical `returns`, for a given `benchmark`. Return values greater than `benchmark` are "upside" returns, values lower than `benchmark` are "downside" returns. Each column of `returns` corresponds to an asset.

Historical semi returns ``\\mathrm{R_b}``, are defined as:

```math
\\mathrm{R_b} = \\dfrac{\\mathrm{R} - b}{\\sqrt{N}}\\,
```

where ``\\mathrm{R}`` are historical returns, ``b`` is the benchmark, and ``N`` is the number of historical entries (not the number of assets). The value of `benchmark` should correspond to the frequency of historical returns, i.e. if using daily returns `benchmark` should be a reasonable value for daily returns.
"""
function semi_ret(returns, benchmark = 0)
    samples = size(returns, 1)
    return (returns .- benchmark) / sqrt(samples)
end
function cdar(alpha, z, samples, beta)
    return alpha + sum(z) / (samples * (1 - beta))
end
function cvar(alpha, u, samples, beta)
    return alpha + sum(u) / (samples * (1 - beta))
end

"""
```
transaction_cost(w, w_prev, k = 0.001)
```

Simple transaction cost model. Sums all the weight absolute changes and multiplies by a fraction `k`. Simulates a fixed percentage commision from a broker.

The transacion cost ``C``, is defined as:

```math
C = k \\lvert \\bm{w} - \\bm{w}_{\\mathrm{prev}} \\rvert
```

!!! warning
    As of JuMP 1.0 there is no support for `norm` in objective functions. A model wishing to add this to their objective function should instead add it as a Vector Cone constraint, eg:
    ```
    n = length(tickers)
    prev_weights = fill(1 / n), n))
    k = 0.001

    extra_vars = [:(z[1:(\$n)])]
    extra_constraints =
        [:([\$k * (model[:w] - \$prev_weights); model[:z]] in MOI.NormOneCone(\$(n + n)))]

    ef = EfficientFrontier(tickers, mu, S;
            extra_vars = extra_vars,
            extra_constraints = extra_constraints,
        )
    ```
    L2-norms must be turned into constraints of type `MOI.NormTwoCones`. More information on how to do this can be found in [JuMP Vector Cones](https://jump.dev/JuMP.jl/stable/moi/manual/standard_form/#Vector-cones).
"""
function transaction_cost(w, w_prev, k = 0.001)
    return k * norm(w - w_prev, 1)
end

"""
```
ex_ante_tracking_error(w, cov_mtx, w_bmk)
```

Compute the square of the ex-ante tracking error from the weights `w`, covariance matrix `cov_mtx`, and benchamark weights `w_bmk`.

The ex-ante tracking error ``\\mathrm{Err}``, is defined as:

```math
\\mathrm{Err} = (\\bm{w} - \\bm{w}_b)^T \\, \\Sigma \\, (\\bm{w} - \\bm{w}_b)\\,,
```

where ``\\bm{w}`` are the weights, ``\\bm{w}_b`` the benchmark weights, and ``\\Sigma`` the covariance matrix.
"""
function ex_ante_tracking_error(w, cov_mtx, w_bmk)
    w_rel = w .- w_bmk
    return dot(w_rel, cov_mtx, w_rel)
end

"""
```
ex_post_tracking_error(w, returns, ret_bmk)
```

Compute the square of the ex-post tracking error from the given weights `w`, historical returns `returns`, and benchmark returns `ret_bmk`.

The ex-post tracking error ``\\mathrm{Err}``, is defined as:

```math
\\begin{aligned}
    x &= \\mathrm{R} \\bm{w} - \\bm{R}_b  \\\\
    \\mu &= \\dfrac{1}{N} \\sum\\limits^N x \\\\
    \\mathrm{Err} &= (x - \\mu) \\cdot (x - \\mu) = Var(r - r_b)\\,,
\\end{aligned}
```

where ``w`` are the weights, ``\\mathrm{R}`` historical returns, ``\\bm{R}_b`` benchmark historical returns. In other words the ex-post tracking error is the variance of the historical returns ``r``, minus the benchmark returns ``r_b``, of the portfolio.
"""
function ex_post_tracking_error(w, returns, ret_bmk)
    x = returns * w .- ret_bmk
    μ = mean(x)
    tmp = (x .- μ)
    return dot(tmp, tmp)
end

"""
```
logarithmic_barrier_objective(w, cov_mtx, k = 0.1)
```

Compute the logarithmic barrier adjusted by `k`, from the given weights `w`, covariance matrix `cov_mtx`.

The logarithmic barrier ``L``, is defined as:

```math
L = \\bm{w}^T\\, \\Sigma\\, \\bm{w} - k \\sum\\limits_i \\ln( w_i )\\,,
```

where ``\\sigma^2 = \\bm{w}^T\\, \\Sigma\\, \\bm{w}`` (see [`port_variance`](@ref)), ``k`` the adjustment constant, and ``w_i`` the weight of the ``i``'th asset.
"""
function logarithmic_barrier_objective(w, cov_mtx, k = 0.1)
    log_sum = sum(log.(w .+ eps()))
    var = dot(w, cov_mtx, w)
    return var - k * log_sum
end

"""
```
kelly_objective(w, mean_ret, cov_mtx, k = 3)
```

Compute a Kelly objective from adjusted by `k`, from the given weights `w`, mean returns `mean_ret`, and covariance matrix `cov_mtx`.

The Kelly objective ``K``, is defined as:

```math
K = \\bm{w}^T\\, \\Sigma\\, \\bm{w} - \\dfrac{k}{2} \\bm{w} \\cdot \\bm{E} \\,,
```

where ``\\sigma^2 = \\bm{w}^T\\, \\Sigma\\, \\bm{w}`` (see [`port_variance`](@ref)), ``k`` is the adjustment constant, ``\\bm{w}`` the weights, and ``\\bm{E}`` the mean returns.
"""
function kelly_objective(w, mean_ret, cov_mtx, k = 3)
    variance = dot(w, cov_mtx, w)
    objective = variance * 0.5 * k - dot(w, mean_ret)
    return objective
end