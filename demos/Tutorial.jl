"""
# PortfolioOptimiser.jl

This is a demo/template for using `PortfolioOptimiser.jl`. It should be all that you need to start to use and adapt the library to your needs.
"""

using PortfolioOptimiser
using CSV, DataFrames, Plots, CovarianceEstimation, LinearAlgebra

"""
## Loading data

All our functions take AbstractArray or Tuple arguments, not DataFrame or TimeArray data. The reason is to keep the code as generic and performant as possible. Julia is fast enough that it can be used in performance critical applications, so we remove the data wrangling from the equation, and we leave it up to the user to decide how they want to handle their data.

For the example we'll use CSV and DataFrames to load our historical prices.
"""

hist_prices = CSV.read("./demos/assets/stock_prices.csv", DataFrame)
dropmissing!(hist_prices)

"""
## Returns

`PortfolioOptimiser.jl` calculates a variety of returns and expected returns. First we have to load the data. In this case, we load daily stock data pertaining to a variety of different tickers into a DataFrame.

The price data is pretty useless until we want to allocate our portfolio, what we want for our calculations need is the return data. We can get both regular and log returns with the function `returns_from_prices`. Log returns are useful because they can be added instead of multiplied when compounding, the function defaults to regular returns.
"""

returns = returns_from_prices(hist_prices[!, 2:end])
log_returns = returns_from_prices(hist_prices[!, 2:end], true)
exp.(log_returns) .- 1 ≈ returns

"""
Whilst we can't recover the exact prices from the returns, we can recover relative prices. As previously stated, we leave data wrangling in the hands of the user, as of the time of writing, DataFrames does not define `cumprod` so we must convert it into an array.
"""

rel_prices = prices_from_returns(Matrix(returns[!, :]))
rel_prices_log = prices_from_returns(Matrix(log_returns[!, :]), true)
rel_prices ≈ rel_prices_log

"""
We can reconstruct the original prices by multiplying the first entry of historical prices by the corresponding relative prices.
"""

reconstructed_prices = (rel_prices' .* Vector(hist_prices[1, 2:end]))'
reconstructed_prices ≈ Matrix(hist_prices[!, 2:end])

"""
## Expected returns

We have a few ways of calculating a variety of expected/mean returns:

- mean
- exponentially weighted mean
- capital asset pricing model
- exponentially weighted capital asset pricing model

The optional keyword arguments are further explained in the docs.

Typically, exponentially-weighted returns are more predictive of future returns, as they assign higher weights to more recent values.
"""

num_rows = nrow(returns)
num_cols = ncol(returns)

past_returns = Matrix(returns[1:div(num_rows, 2), :])
future_returns = Matrix(returns[(div(num_rows, 2) + 1):end, :])

mean_future_rets = ret_model(MRet(), future_returns)

mean_ret = ret_model(MRet(), past_returns)
exp_mean_ret = ret_model(EMRet(), past_returns, span = num_rows / 2)
capm_ret = ret_model(CAPMRet(), past_returns)
exp_capm_ret =
    ret_model(ECAPMRet(), past_returns, cspan = num_rows / 2, rspan = num_rows / 2)

"""
Using [CovarianceEstimation.jl](https://github.com/mateuszbaran/CovarianceEstimation.jl) and CustomCov() or CustomSCov() (see Risk Models section), we can come up with some more robust measurements of CAPM returns.

Downside covariance tends to be more stable, to prove this we use both the shrunken covariance and shrunken semicovariance. Note that we can use any covariance type by simply changing the `cov_type` keyword argument. The default is to use the sample covariance for CAPMRet(), and exponentially weighted covariance for ECAPMRet().
"""

target = DiagonalCommonVariance()
shrinkage = :oas
method = LinearShrinkage(target, shrinkage)

capm_ret_shrunken_cov = ret_model(
    CAPMRet(),
    past_returns,
    cov_type = CustomCov(),
    custom_cov_estimator = method,
)

exp_capm_ret_shrunken_cov = ret_model(
    ECAPMRet(),
    past_returns,
    cspan = num_rows / 2,
    rspan = num_rows / 2,
    cov_type = CustomCov(),
    custom_cov_estimator = method,
)

capm_ret_shrunken_scov = ret_model(
    CAPMRet(),
    past_returns,
    cov_type = CustomSCov(),
    custom_cov_estimator = method,
)

exp_capm_ret_shrunken_scov = ret_model(
    ECAPMRet(),
    past_returns,
    cspan = num_rows / 2,
    rspan = num_rows / 2,
    cov_type = CustomSCov(),
    custom_cov_estimator = method,
)

errors = Float64[]
push!(errors, sum(abs.(mean_future_rets - mean_ret)))
push!(errors, sum(abs.(mean_future_rets - exp_mean_ret)))
push!(errors, sum(abs.(mean_future_rets - capm_ret)))
push!(errors, sum(abs.(mean_future_rets - capm_ret_shrunken_cov)))
push!(errors, sum(abs.(mean_future_rets - capm_ret_shrunken_scov)))
push!(errors, sum(abs.(mean_future_rets - exp_capm_ret)))
push!(errors, sum(abs.(mean_future_rets - exp_capm_ret_shrunken_cov)))
push!(errors, sum(abs.(mean_future_rets - exp_capm_ret_shrunken_scov)))

errors /= length(mean_future_rets)

fig = plot(
    ["M", "EM", "CAPM", "CAPM S C", "CAPM S SC", "ECAPM", "ECAPM S C", "ECAPM S SC"],
    errors,
    ylabel = "Rel err",
    legend = false,
    size = (700, 400),
    seriestype = :bar,
)

println(errors)

"""
Returns are chaotic and unpredictable, so it's often better to optimise portfolios without considering returns. The average absolute errors are all over 30%, so for this case, a portfolio that has an expected return of 10 %, will most likely return between [-20, 40] %.

Minimum volatility, semivariance, CVaR and CDaR tend to give more stable portfolios than ones that take returns into consideration.

We can plot all the return types together to see how they correlate to each other.
"""
l = @layout [a b c d; e f g h]

fig1 = bar(
    mean_ret,
    yticks = (1:num_cols, names(returns)),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "M",
)

fig2 = bar(
    exp_mean_ret,
    yticks = (1:num_cols, []),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "EM",
)

fig3 = bar(
    capm_ret,
    yticks = (1:num_cols, []),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "CAPM",
)

fig4 = bar(
    capm_ret_shrunken_cov,
    yticks = (1:num_cols, []),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "CAPM S C",
)

fig5 = bar(
    capm_ret_shrunken_scov,
    yticks = (1:num_cols, names(returns)),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "CAPM S SC",
)

fig6 = bar(
    exp_capm_ret,
    yticks = (1:num_cols, []),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "ECAPM",
)

fig7 = bar(
    exp_capm_ret_shrunken_cov,
    yticks = (1:num_cols, []),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "ECAPM S C",
)

fig8 = bar(
    exp_capm_ret_shrunken_scov,
    yticks = (1:num_cols, []),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "ECAPM S SC",
)

plot(fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, layout = l, size = (1200, 1200))

"""
There's strong correspondence between exponential and non-exponentially weighted return types. CAPM returns are a little bit different as they account for an asset's relationship to the market via a covariance matrix.
"""

"""
## Risk models

We also have a few built-in risk models. However, using CustomCov() and CustomSCov() we can make use of other models, such as those found in [CovarianceEstimation.jl](https://github.com/mateuszbaran/CovarianceEstimation.jl), and use other types of weights other than exponential ones.

We provide a variety of covariance measures:

- sample covariance
- exponentially weighted sample covariance
- semicovariance
- exponentially weighted semicovariance
- custom covariance
- custom semicovariance

The optional keyword arguments are further explained in the docs.

We can see which covariance matrix estimates the future asset variances the best.
"""

future_cov = cov(Cov(), future_returns)
future_semi_cov = cov(SCov(), future_returns)

sample_cov = cov(Cov(), past_returns)
exp_cov = cov(ECov(), past_returns, span = num_rows / 2)
semi_cov = cov(SCov(), past_returns)
exp_semi_cov = cov(ESCov(), past_returns, span = num_rows / 2)

target = DiagonalCommonVariance()
shrinkage = :oas
method = LinearShrinkage(target, shrinkage)

oas_shrunken_cov = Matrix(cov(CustomCov(), past_returns, estimator = method))
oas_shrunken_cov_semi_cov = Matrix(cov(CustomSCov(), past_returns, estimator = method))

future_var = diag(future_cov)
future_semivar = diag(future_semi_cov)

errors = Float64[]
push!(errors, sum(abs.(future_var - diag(sample_cov))))
push!(errors, sum(abs.(future_var - diag(exp_cov))))
push!(errors, sum(abs.(future_var - diag(oas_shrunken_cov))))
push!(errors, sum(abs.(future_semivar - diag(semi_cov))))
push!(errors, sum(abs.(future_semivar - diag(exp_semi_cov))))
push!(errors, sum(abs.(future_semivar - diag(oas_shrunken_cov_semi_cov))))

errors /= length(future_var)

fig = bar(
    ["Sample", "Exp Sample", "OAS", "Semi", "Exp Semi", "OAS Semi"],
    errors,
    ylabel = "Rel err",
    legend = false,
)

println(errors)

"""
This is a toss up, as the non exponentially weighted covariances did a better job of estimating the correlations between assets than the corresponding exponentially weighted ones.

Furthermore, this lends more weight to the concept of using optimisations which account for downside risk such as EffMeanSemivar, CVaR and CDaR optimisations, and using the semicovariance as the input covariance matrix for Black-Litterman and Hierarchical Risk Parity optimisations.

We can do the same for the covariances, which is what the optimisations actually use.
"""

errors = Float64[]
push!(errors, sum(abs.(future_cov - sample_cov)))
push!(errors, sum(abs.(future_cov - exp_cov)))
push!(errors, sum(abs.(future_semi_cov - semi_cov)))
push!(errors, sum(abs.(future_semi_cov - exp_semi_cov)))

errors /= length(future_cov)

println(errors)

fig = bar(
    ["Sample", "Exp Sample", "Semi", "Exp Semi"],
    errors,
    ylabel = "Rel err",
    legend = false,
)

"""
Again, this is a bit of a toss up between choosing exponentially weighted vs not, but the estimation error is again lower when using semicovariances.

If we plot the absolute errors of the correlation matrices we can see that the semicovariance is a more stable measure of the correlation between assets, note how much darker the second plot is compared to the first.
"""

fig1 = heatmap(
    abs.(cov2cor(future_cov) - cov2cor(sample_cov)) / length(future_cov),
    yflip = true,
    xticks = (1:num_cols, names(returns)),
    yticks = (1:num_cols, names(returns)),
    xrotation = 70,
    clims = (0, 1e-3),
)

fig2 = heatmap(
    abs.(cov2cor(future_semi_cov) - cov2cor(semi_cov)) / length(future_semi_cov),
    yflip = true,
    xticks = (1:num_cols, names(returns)),
    yticks = (1:num_cols, names(returns)),
    xrotation = 70,
    clims = (0, 1e-3),
)

"""
# Efficient Frontier Optimisation

First we import the packages we need. In this case we will download the data using MarketData.
"""

using PortfolioOptimiser, DataFrames, MarketData

"""
We will use the following tickers as they tend to be trendy.
"""

tickers = sort!([
    "MSFT",
    "AMZN",
    "KO",
    "MA",
    "COST",
    "LUV",
    "XOM",
    "PFE",
    "JPM",
    "UNH",
    "ACN",
    "DIS",
    "GILD",
    "F",
    "TSLA",
])

"""
Download the data from yahoo finance, and obtain the historical prices for the tickers.
"""

data = yahoo.(tickers)
hist_prices = merge([data[i][:Close] for i in 1:length(data)]...)
hist_prices = DataFrames.rename(DataFrame(hist_prices), ["timestamp"; tickers])

"""
Plot the data to see what it looks like.
"""

fig = plot()
for ticker in tickers
    plot!(fig, hist_prices[!, 1], hist_prices[!, ticker], label = ticker)
end
plot(fig, legend = :topleft)

"""
Calculate the returns and sample covariance.
"""

returns = returns_from_prices(hist_prices[!, 2:end])
num_cols = ncol(returns)
num_rows = nrow(returns)

sample_cov = cov(SCov(), Matrix(returns), freq = 252)

fig1 = heatmap(
    cov2cor(sample_cov),
    yflip = true,
    xticks = (1:num_cols, names(returns)),
    yticks = (1:num_cols, names(returns)),
    xrotation = 70,
)

"""
The sample covariance is often not the best, especially when the number of tickers is greater than the number of observations. It can also capture some extreme values, causing the optimiser to overfit the model.

The package [CovarianceEstimation.jl](https://github.com/mateuszbaran/CovarianceEstimation.jl) implements methods to estimate more robust measures of the covariance. Here we use it for our purposes.
"""

using CovarianceEstimation

target = DiagonalUnequalVariance()
shrinkage = :lw
method = LinearShrinkage(target, shrinkage)

S = cov(CustomCov(), Matrix(returns), estimator = method)

fig1 = heatmap(
    cov2cor(Matrix(S)),
    yflip = true,
    xticks = (1:num_cols, names(returns)),
    yticks = (1:num_cols, names(returns)),
    xrotation = 70,
)

"""
## Mean Variance optimisation.

As previously stated, using returns is not the best idea given how volatile they can be. Returns are optional on portfolios that do not use them such as `min_risk!`.

As we are demonstrating all Mean Variance optimisations we will use returns. We'll use exponentially weighted CAPM returns with the LeDoit-Wolf covariance shrinkage as defined above as our mean returns. This is because CAPM returns aim to be more stable than mean historical returns, combining them with a shrunken covariance and assigning exponentially increasing weights to more recent entries should give a better indication of future returns.
"""

capm_ret = ret_model(
    CAPMRet(),
    Matrix(returns),
    cov_type = CustomCov(),
    custom_cov_estimator = method,
)

exp_capm_ret = ret_model(
    ECAPMRet(),
    Matrix(returns),
    cspan = num_rows / 2,
    rspan = num_rows / 2,
    cov_type = CustomCov(),
    custom_cov_estimator = method,
)

[tickers exp_capm_ret]

fig = bar(
    exp_capm_ret,
    yticks = (1:num_cols, tickers),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "Exp CAPM",
)

"""
We can optimise for the minimum volatility without providing expected returns. In this example we also allow the weight bounds to be between -1 and 1, because we want to create a long-short portfolio.
"""

ef = EffMeanVar(tickers, nothing, S, weight_bounds = (-1, 1))
min_risk!(ef)
portfolio_performance(ef, verbose = true)
[tickers ef.weights]

"""
However, if we want to optimise for other objectives in Mean-Variance optmisation we must provide the expected returns. In this example we also allow the weights to be negative such that we can optimise 
"""

ef = EffMeanVar(tickers, exp_capm_ret, S, weight_bounds = (-1, 1))
min_risk!(ef)
portfolio_performance(ef, verbose = true)
[tickers ef.weights]

"""
Assuming we want to stick with this portfolio we then need to allocate it depending on how much money we have. Lets pretend we want a long-short portfolio with a ratio of 169:69, long:short and we have 420,000 to invest. We can do this with mixed-integer optimisation for discrete shares, or a greedy algorithm that supports fractional shares.
"""

alloc, leftover = Allocation(
    LP(),
    ef,
    Vector(hist_prices[end, 2:end]);
    investment = 420_000,
    short_ratio = 0.1,
)

[alloc.tickers alloc.shares]

"""
We can see that the weights between the optimal and allocated portfolios are not the same, because we wanted a specific short:long ratio, we have a finite amount of money, and we are restricted to whole shares by the mixed-integer optimisation.
"""

[alloc.weights[sortperm(alloc.tickers)] ef.weights]

"""
If we remove the short ratio requirement we get something different.
"""

alloc, leftover =
    Allocation(LP(), ef, Vector(hist_prices[end, 2:end]); investment = 420_000)

[alloc.tickers alloc.shares]

[alloc.weights[sortperm(alloc.tickers)] ef.weights]

"""
All four efficient frontier optimisers have five predefined objectives, `min_risk!`, `max_sharpe!`, `max_utility!`, `efficient_return!` and `efficient_risk!`. On top of these, there are two optimisers that let you use custom objective functions. Here we illustrate how you can use both to produce the same optimisation, and it's a good show of how providing returns can lead optimisations astray.
"""

# Use some type of CAPM returns for more stable results.
mean_ret = ret_model(MRet(), Matrix(returns))
tickers = names(hist_prices[!, 2:end])

"""
Use the built-in function that maximises the sharpe ratio (ratio between returns and risk measure). It uses a special variable transformation to turn the ratio into a constrained convex optimisation. It means adding objective functions may not have the desired effcect. In case extra objective terms are needed, it's best to use the nonlinear optimiser.
"""

l_cvar = EffCVaR(tickers, mean_ret, Matrix(returns))
max_sharpe!(l_cvar)
mu, risk = portfolio_performance(l_cvar, verbose = true)

"""
Now we can use the `custom_nloptimiser!` optimise the ratio explicitly. First we need to create the EffCVaR instance.
"""

nl_cvar = EffCVaR(tickers, mean_ret, Matrix(returns))

"""
We need to define the extra optimisation variables (JuMP variables), as well as parameters (non-JuMP variables). We start with the model variables.
"""

model = nl_cvar.model

# The weights are always needed as optimisation variables, so they don't need to be in the extra variables vector.
w = model[:w]
# Alpha, in this case this stands for the cvar variable.
alpha = model[:alpha]
# u is the value at risk optimisation variable.
u = model[:u]

"""
For non-linear models, we often have to provide initial guesses for optimisation variables. Especially if a variable appears in a denominator, this avoids division by zero in the first iteration. We do this as a tuple, the extra variables have to be a collection of 2-tuples: `extra_vars = [(var1, init_val1), (var2, init_val2), ...]` or `extra_vars = ((var1, init_val1), (var2, init_val2), ...)`
"""

extra_vars = [(alpha, nothing), (u, 1 / length(u))]

"""
The model parameters are just regular variables.
"""

mean_ret = nl_cvar.mean_ret
beta = nl_cvar.beta
rf = nl_cvar.rf

obj_params = [length(w), mean_ret, beta, rf]

function nl_cvar_sharpe(w...)
    n = obj_params[1]
    mean_ret = obj_params[2]
    beta = obj_params[3]
    rf = obj_params[4]

    weights = w[1:n]
    alpha = w[n + 1]
    u = w[(n + 2):end]

    samples = length(u)

    ret = PortfolioOptimiser.port_return(weights, mean_ret) - rf
    CVaR = PortfolioOptimiser.cvar(alpha, u, samples, beta)

    return -ret / CVaR
end
custom_nloptimiser!(nl_cvar, nl_cvar_sharpe, obj_params, extra_vars)

"""
Note how the weights are similar up to five parts in a hundred thousand.
"""

isapprox(l_cvar.weights, nl_cvar.weights, rtol = 5e-5)

"""
However, if you get the performance. The cvar is much lower, and the ratio much greater. This is because the non-linear optimiser's tolerance applies to the whole non-linear system, rather than the transformed objective function. However, the weights are much the same, which lends credence to the fact that using returns can lead optimisers to over-fit models, and a small change can lead to a large difference in "performance".
"""

nl_mu, nl_risk = portfolio_performance(nl_cvar, verbose = true)

"""
If we minimise the risk, we can see that even though the max ratio is maximised, the previous cases, neither minimise the CVaR.
"""

l_cvar = EffCVaR(tickers, mean_ret, Matrix(returns))
min_risk!(l_cvar)
min_mu, min_risk = portfolio_performance(l_cvar, verbose = true)

"""
We can do the same with more stable returns.
"""

num_rows = nrow(returns)

target = DiagonalCommonVariance()
shrinkage = :oas
method = LinearShrinkage(target, shrinkage)

mean_ret = ret_model(
    ECAPMRet(),
    Matrix(returns),
    cspan = num_rows,
    rspan = num_rows,
    cov_type = CustomCov(),
    custom_cov_estimator = method,
)

"""
Using the variable transformation.
"""

l_cvar = EffCVaR(tickers, mean_ret, Matrix(returns))
max_sharpe!(l_cvar)
mu, risk = portfolio_performance(l_cvar, verbose = true)

"""
Using the custom nonlinear optimiser.
"""

nl_cvar = EffCVaR(tickers, mean_ret, Matrix(returns))

model = nl_cvar.model
w = model[:w]
alpha = model[:alpha]
u = model[:u]
extra_vars = [(alpha, nothing), (u, 1 / length(u))]

mean_ret = nl_cvar.mean_ret
beta = nl_cvar.beta
rf = nl_cvar.rf

obj_params = [length(w), mean_ret, beta, rf]

custom_nloptimiser!(nl_cvar, nl_cvar_sharpe, obj_params, extra_vars)
nl_mu, nl_risk = portfolio_performance(nl_cvar, verbose = true)

"""
Note how in both cases, max_sharpe! optimises to a similar value of CVaR, but using ECAPM returns with a shrunken covariance yields an of the CVaR that is closer to the minimum possible.

Thankfully, despite the wildly different ratios between optimisers, the weights are ultimately similar up to five parts in then thousand.
"""

isapprox(l_cvar.weights, nl_cvar.weights, rtol = 5e-4)

######################################

cvar = EffCVaR(tickers, mean_ret, Matrix(returns))
min_risk!(cvar)
portfolio_performance(cvar, verbose = true)

cdar = EffCDaR(tickers, mean_ret, Matrix(returns))
max_sharpe!(cdar)
mu, risk = portfolio_performance(cdar, verbose = true)

nl_cdar = EffCDaR(tickers, mean_ret, Matrix(returns))
model = nl_cdar.model
alpha = model[:alpha]
z = model[:z]
w = model[:w]
mean_ret = nl_cdar.mean_ret
beta = nl_cdar.beta
rf = nl_cdar.rf

extra_vars = [(alpha, nothing), (z, 1 / length(z))]
obj_params = [length(w), mean_ret, beta, rf]

function nl_cdar_sharpe(w...)
    n = obj_params[1]
    mean_ret = obj_params[2]
    beta = obj_params[3]
    rf = obj_params[4]

    weights = w[1:n]
    alpha = w[n + 1]
    z = w[(n + 2):end]

    samples = length(z)

    ret = PortfolioOptimiser.port_return(weights, mean_ret) - rf
    CDaR = PortfolioOptimiser.cdar(alpha, z, samples, beta)

    return -ret / CDaR
end
custom_nloptimiser!(nl_cdar, nl_cdar_sharpe, obj_params, extra_vars)
isapprox(cdar.weights, nl_cdar.weights, rtol = 1e-4)
nl_mu, nl_risk = portfolio_performance(nl_cdar, verbose = true)

cdar = EffCDaR(tickers, mean_ret, Matrix(returns))
min_risk!(cdar)
mu, risk = portfolio_performance(cdar, verbose = true)
