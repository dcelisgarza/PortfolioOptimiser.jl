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
sharpe_ratio(w, mean_ret, cov_mtx, rf::Real = 1.02^(1/252)-1)
```

Computes the portfolio's Sharpe ratio given the weights `w`, mean returns `mean_ret`, covariance matrix `cov_mtx`, and risk free rate `rf`.

```
sharpe_ratio(μ, σ, rf::Real = 1.02^(1/252)-1)
```

Computes the Sharpe ratio given the return `μ`, standard deviation `σ`, and risk free rate `rf`.

The Sharpe ratio ``s``, is defined as:

```math
s = \\dfrac{\\mu - r}{\\sigma},
```

where ``\\mu`` the portfolio's return (see [`port_return`](@ref)), and ``\\sigma`` the portfolio's standard deviation (see [`port_variance`](@ref)). Generally speaking, the greater the Sharpe ratio the better the portfolio.

!!! note
    The Sharpe ratio penalises large swings in both directions, so assets that tend to have large increases in value are disproportionally penalised by this measure. The Sortino ratio has the same formula but uses an adjusted covariance matrix that accounts only for the negative fluctuations in value. The semicovariance is implemented by [`cov`](@ref) when given `SCov()` or `ESCov()` as its first argument. The Mean-Semivariance optimisations [`EffMeanSemivar`](@ref) make the adjustment too.
"""
function sharpe_ratio(w, mean_ret, cov_mtx, rf::Real = 1.02^(1 / 252) - 1)
    μ = port_return(w, mean_ret)
    σ = sqrt(port_variance(w, cov_mtx))
    return (μ - rf) / σ
end
function sharpe_ratio(μ, σ, rf::Real = 1.02^(1 / 252) - 1^(1 / 252) - 1)
    return (μ - rf) / σ
end

"""
```
port_semivar(w, returns, target = 0, freq = 252)
```

Compute the semivariance from the weights `w`, historical returns `returns`, according to the target `target`, and frequency of returns `freq`.

The semivariance is defined as:

```math
\\begin{aligned}
\\bm{r} &= \\mathrm{R} \\bm{w}\\,,\\\\
\\sigma_s^2 &= \\dfrac{f}{N} \\sum\\limits_{i = 1,\\, b < r_i}^{N} (r_i - b)^2\\,,
\\end{aligned}
```

where ``\\bm{r}`` are the portfolio historical returns with where the subscript ``i`` describes a specific point in time (entry), ``\\mathrm{R}`` the asset historical returns, ``\\bm{w}`` the asset weights, ``\\sigma_s`` the semideviation, ``f`` the frequency of the historical returns, ``N`` the number of entries in historical returns (not the number of assets), and ``b`` the target for splitting "upside" and "downside" returns.

The condition ``b < r_i`` ensures we only consider entries for which the historical portfolio return fell below the target.
"""
function port_semivar(w, returns, target = 0, freq = 252)
    port_ret = returns * w
    port_ret = min.(port_ret .- target, 0)

    return dot(port_ret, port_ret) / size(returns, 1) * freq
end

function port_mean_abs_dev(w, returns, target = zeros(length(w)), freq = 252)
    port_ret = returns * w
    port_target = dot(target, w)
    return sum(port_ret .- port_target) * freq / size(returns, 1)
end

"""
```
L2_reg(w, γ = 1)
```

L2 regularisation. Minimising this reduces the number of negligible weights `w`, with a tuning parameter `γ`. Increasing γ will decrease the number of negligible weights.

The ``L_2`` regularisation is defined as:

```math
L_2 = \\gamma (\\bm{w} \\cdot \\bm{w}) =  \\gamma \\lVert \\bm{w} \\rVert ^2\\,,
```

where ``\\gamma`` is the tuning parameter, and ``\\bm{w}`` the weights.
"""
function L2_reg(w, γ = 1)
    return γ * dot(w, w)
end

"""
```
quadratic_utility(w, mean_ret, cov_mtx, risk_aversion = 1)
```

Calculates the quadratic utility given the weights `w`, mean returns `mean_ret`, covariance matrix `cov_mtx` and risk aversion `risk_aversion`. Increasing `risk_aversion` decreases risk. 

The quadratic utility ``Q``, is defined as:

```math
Q = \\mu - \\dfrac{1}{2} \\delta  \\bm{w}^T\\, \\Sigma\\, \\bm{w}\\,,
```

where ``\\mu`` is the overall portfolio expected return (see [`port_return`](@ref)), ``\\delta`` the risk aversion, ``\\bm{w}`` the weights, and ``\\Sigma`` the covariance matrix.
"""
function quadratic_utility(w, mean_ret, cov_mtx, risk_aversion = 1)
    μ = port_return(w, mean_ret)
    σ2 = port_variance(w, cov_mtx)
    return μ - 0.5 * risk_aversion * σ2
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

Simple transaction cost model. Sums all the absolute changes in weight and multiplies by a constant `k`. Simulates a fixed percentage commision from a broker.

The transacion cost ``C``, is defined as:

```math
C = k \\lvert \\bm{w} - \\bm{w}_{\\mathrm{prev}} \\rvert\\,,
```

where ``k`` is the fixed percentage commision, ``\\bm{w}`` the asset weights, and ``\\bm{w}_{\\mathrm{prev}}`` the previous weights of the assets.

!!! warning
    JuMP doesn't yet support norm in the objective (v1.0), so we need to turn them into a variable subject to a [MOI.NormOneCone](https://docs.juliahub.com/MathOptInterface/tyub8/1.3.0/reference/standard_form/#MathOptInterface.NormOneCone) constraint. We can follow the example in the [JuMP tutorial](https://jump.dev/JuMP.jl/stable/tutorials/conic/logistic_regression/#\\ell_1-regularized-logistic-regression) to do this.

    ```julia
    n = length(tickers)
    prev_weights = fill(1 / n, n)
    k = 0.001
    ef = EffMeanVar(
        tickers,
        mu,
        S;
        # Add the variable that will contain the value of the l1 regularisation.
        extra_vars = [:(0 <= l1)],
        # We constrain it to be within a MOI.NormOneCone.
        extra_constraints = [
            :([model[:l1]; (model[:w] - \$prev_weights)] in MOI.NormOneCone(\$(n + 1))),
        ],
        # We add the variable to the objective and multiply it by the adjustment parameter.
        extra_obj_terms = [quote
            \$k * model[:l1]
        end],
    )
    ```
    Similarly L2-norms must be turned into constraints of type `MOI.NormTwoCone`. More information on how to do this can be found in [JuMP Vector Cones](https://jump.dev/JuMP.jl/stable/moi/manual/standard_form/#Vector-cones).
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

where ``\\bm{w}`` are the weights, ``\\bm{w}_b`` the target weights, and ``\\Sigma`` the covariance matrix.
"""
function ex_ante_tracking_error(w, cov_mtx, w_bmk)
    w_rel = w .- w_bmk
    return dot(w_rel, cov_mtx, w_rel)
end

"""
```
ex_post_tracking_error(w, returns, ret_bmk)
```

Compute the square of the ex-post tracking error from the weights `w`, historical returns `returns`, and target returns `ret_bmk`.

The ex-post tracking error ``\\mathrm{Err}``, is defined as:

```math
\\begin{aligned}
    \\bm{x} &= \\mathrm{R} \\bm{w} - \\bm{R}_b  \\,,\\\\
    \\mu &= \\dfrac{1}{N} \\sum\\limits_{i=1}^N x_i \\,,\\\\
    \\mathrm{Err} &= (\\bm{x} - \\mu) \\cdot (\\bm{x} - \\mu) = Var(r - r_b)\\,,
\\end{aligned}
```

where ``w`` are the weights, ``\\mathrm{R}`` historical returns, ``\\bm{R}_b`` target historical returns. In other words the ex-post tracking error is the variance of the historical returns ``r``, minus the target historical returns ``r_b``, of the portfolio.
"""
function ex_post_tracking_error(w, returns, ret_bmk)
    x = returns * w - ret_bmk
    μ = mean(x)
    tmp = (x .- μ)
    return dot(tmp, tmp)
end

"""
```
logarithmic_barrier(w, cov_mtx, k = 0.1)
```

Compute the logarithmic barrier adjusted by `k`, from the weights `w`, and covariance matrix `cov_mtx`.

The logarithmic barrier ``L``, is defined as:

```math
L = \\bm{w}^T\\, \\Sigma\\, \\bm{w} - k \\sum\\limits_i \\ln( w_i )\\,,
```

where ``\\sigma^2 = \\bm{w}^T\\, \\Sigma\\, \\bm{w}`` (see [`port_variance`](@ref)), ``k`` the adjustment constant, and ``w_i`` the weight of the ``i``'th asset.
"""
function logarithmic_barrier(w, cov_mtx, k = 0.1)
    # Add eps() to avoid log(0) divergence.
    log_sum = sum(log.(w .+ eps()))
    var = dot(w, cov_mtx, w)
    return var - k * log_sum
end

"""
```
kelly_objective(w, mean_ret, cov_mtx, k = 3)
```

Compute a Kelly objective from adjusted by `k`, from the weights `w`, mean returns `mean_ret`, and covariance matrix `cov_mtx`.

The Kelly objective ``K``, is defined as:

```math
K = \\bm{w}^T\\, \\Sigma\\, \\bm{w} - \\dfrac{1}{2} k \\bm{w} \\cdot \\bm{E} \\,,
```

where ``\\sigma^2 = \\bm{w}^T\\, \\Sigma\\, \\bm{w}`` (see [`port_variance`](@ref)), ``k`` is the adjustment constant, ``\\bm{w}`` the weights, and ``\\bm{E}`` the mean returns.
"""
function kelly_objective(w, mean_ret, cov_mtx, k = 3)
    variance = dot(w, cov_mtx, w)
    objective = 0.5 * variance * k - dot(w, mean_ret)
    return objective
end