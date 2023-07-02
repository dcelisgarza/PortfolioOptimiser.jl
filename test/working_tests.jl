using PortfolioOptimiser, DataFrames, TimeSeries, Dates

println(fieldnames(Portfolio))

A = TimeArray(collect(Date(2023, 03, 01):Day(1):(Date(2023, 03, 20))), rand(20, 10))
Y = percentchange(A)
@time test = Portfolio(returns = DataFrame(Y), short = false, sum_short_long = 1)
isinf(test.upper_average_drawdown)