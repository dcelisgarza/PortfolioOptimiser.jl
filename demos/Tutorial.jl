"""
# PortfolioOptimiser.jl

This is a demo/template for using `PortfolioOptimiser.jl`. It should be all that you need to start to use and adapt the library to your needs.
"""

using PortfolioOptimiser
using CSV, DataFrames, Plots

"""
## Loading data

All our functions take AbstractArray or Tuple arguments, not DataFrame or TimeArray data. The reason is to keep the code as generic and performant as possible. Julia is fast enough that it can be used in performance critical applications, so we remove the data wrangling from the equation, and we leave it up to the user to decide how they want to handle their data.

For the example we'll use CSV and DataFrames to load our historical prices.
"""

hist_prices = CSV.read("./assets/stock_prices.csv", DataFrame)
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

errors = Float64[]
push!(errors, sum(abs.(mean_future_rets - mean_ret)))
push!(errors, sum(abs.(mean_future_rets - exp_mean_ret)))
push!(errors, sum(abs.(mean_future_rets - capm_ret)))
push!(errors, sum(abs.(mean_future_rets - exp_capm_ret)))

errors /= length(mean_future_rets)

fig = bar(
    ["Mean", "Exp Mean", "CAPM", "Exp CAPM"],
    errors,
    ylabel = "Rel err",
    legend = false,
)

println(errors)

"""
Returns are chaotic and unpredictable, so it's often better to optimise portfolios without considering returns. The average absolute errors are all over 30%, so for this case, a portfolio that has an expected return of 10 %, will most likely return between [-20, 40] %.

Minimum volatility, semivariance, CVaR and CDaR tend to give more stable portfolios than ones that take returns into consideration.

We can plot all the return types together to see how they correlate to each other.
"""
l = @layout [a b c d]

fig1 = bar(
    mean_ret,
    yticks = (1:num_cols, names(returns)),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "Mean",
)

fig2 = bar(
    exp_mean_ret,
    yticks = (1:num_cols, []),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "Exp Mean",
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
    exp_capm_ret,
    yticks = (1:num_cols, []),
    orientation = :h,
    yflip = true,
    legend = false,
    title = "Exp CAPM",
)

plot(fig1, fig2, fig3, fig4, layout = l, size = (1000, 500))

"""
There's strong correspondence between exponential and non-exponentially weighted return types. CAPM returns are a little bit different as they account for an asset's relationship to the market via a covariance matrix.
"""

"""
## Risk models

We also have a few built-in risk models. For more models see [CovarianceEstimation.jl](https://github.com/mateuszbaran/CovarianceEstimation.jl).

We provide a variety of covariance measures:

- sample covariance
- exponentially weighted sample covariance
- semicovariance
- exponentially weighted semicovariance

The optional keyword arguments are further explained in the docs.

We can see which covariance matrix estimates the future asset variances the best.
"""

future_cov = risk_model(Cov(), future_returns)
future_semi_cov = risk_model(SCov(), future_returns)

sample_cov = risk_model(Cov(), past_returns)
exp_cov = risk_model(ECov(), past_returns, span = num_rows / 2)
semi_cov = risk_model(SCov(), past_returns)
exp_semi_cov = risk_model(ESCov(), past_returns, span = num_rows / 2)

future_var = diag(future_cov)
future_semivar = diag(future_semi_cov)

errors = Float64[]
push!(errors, sum(abs.(future_var - diag(sample_cov))))
push!(errors, sum(abs.(future_var - diag(exp_cov))))
push!(errors, sum(abs.(future_semivar - diag(semi_cov))))
push!(errors, sum(abs.(future_semivar - diag(exp_semi_cov))))

errors /= length(future_var)

fig = bar(
    ["Sample", "Exp Sample", "Semi", "Exp Semi"],
    errors,
    ylabel = "Rel err",
    legend = false,
)

println(errors)

"""
This is a toss up, as the non exponentially weighted covariances did a better job of estimating the correlations between assets than the corresponding exponentially weighted ones.

Furthermore, this lends more weight to the concept of using optimisations which account for downside risk such as MeanSemivar, CVaR and CDaR optimisations, and using the semicovariance as the input covariance matrix for Black-Litterman and Hierarchical Risk Parity optimisations.

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
using PortfolioOptimiser, DataFrames, MarketData, TimeArray, CovarianceEstimation

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

sample_cov = risk_model(SCov(), Matrix(returns), freq = 252)

fig1 = heatmap(
    cov2cor(S),
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
lw_shrunken2 = cov(method, Matrix(returns)) * 252

fig1 = heatmap(
    cov2cor(Matrix(S)),
    yflip = true,
    xticks = (1:num_cols, names(returns)),
    yticks = (1:num_cols, names(returns)),
    xrotation = 70,
)

capm_ret =
    ret_model(CAPMRet(), Matrix(returns), custom_cov = cov, custom_cov_estimator = method)
exp_capm_ret = ret_model(
    ECAPMRet(),
    Matrix(returns),
    cspan = num_rows / 2,
    rspan = num_rows / 2,
    custom_cov = cov,
    custom_cov_estimator = method,
)