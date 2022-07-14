using JuMP, Ipopt

m = Model(Ipopt.Optimizer)
@variable(m, x >= 0)
@NLexpression(m, risk, 3x^3 - 2x^2 + 5x + 5)
@NLobjective(m, Min, risk)
optimize!(m)
objective_value(m)

# Use this to add terms to nonlinear objectives and nonlinear expressions.
m = Model(Ipopt.Optimizer)
@variable(m, x >= 0)
@NLexpression(m, risk, 3x^3 - 2x^2 + 5x + 5)

# Add to expression that will become the objective before making the objective.
wak = add_nonlinear_expression(m, :($(m[:risk]) + log($(m[:x]) * 3)))
unregister(m, :risk)
@NLexpression(m, risk, wak)
@NLobjective(m, Min, risk)
optimize!(m)
objective_value(m)
print(m)

m = Model(Ipopt.Optimizer)
@variable(m, x >= 0)
@NLexpression(m, risk, 17exp(11x^3) - log(7x^2) + sin(13x) + 5)

@NLobjective(m, Min, risk)
println(fieldnames(typeof(m.nlp_data)))
# Nonlinear expressions are here, probably shouldn't delete them.
m.nlp_data.nlexpr
fieldnames(typeof(m.nlp_data.nlexpr[1]))
fieldnames(typeof(m.nlp_data.nlexpr[1].nd[1]))
m.nlp_data.nlexpr[1].nd
m.nlp_data.nlexpr[1].nd
m.nlp_data.nlexpr[1].const_values
m.nlp_data.nlobj

m.nlp_data.nlobj.nd[1]
m.nlp_data.nlobj.const_values

risk = add_nonlinear_expression(m, :(exp($(x))))

risk
# println(fieldnames(typeof(m)))
# display(m.nlp_data)

using PortfolioOptimiser, DataFrames
using Plots, CovarianceEstimation
using LinearAlgebra, JuMP, IJulia
using CSV, MarketData, Statistics

hist_prices = CSV.read("./demos/assets/stock_prices.csv", DataFrame)
dropmissing!(hist_prices)

returns = returns_from_prices(hist_prices[!, 2:end])

target = DiagonalCommonVariance()
shrinkage = :oas
method = LinearShrinkage(target, shrinkage)

S = cov(CustomCov(), Matrix(returns), estimator = method)

mean_ret = ret_model(
    ECAPMRet(),
    Matrix(returns),
    # cspan = num_rows,
    # rspan = num_rows,
    cov_type = CustomCov(),
    custom_cov_estimator = method,
)

using ECOS
tickers = names(hist_prices[!, 2:end])

emv = EffMeanVar(tickers, mean_ret, S)
@time max_sharpe!(emv, optimiser = ECOS.Optimizer)

w1 = copy(emv.weights)
obj1 = objective_value(emv.model)

emv = EffMeanVar(tickers, mean_ret, S)
@time max_sharpe!(emv, optimiser = ECOS.Optimizer)
w2 = copy(emv.weights)
obj2 = objective_value(emv.model)

emv = EffMeanVar(tickers, mean_ret, S)
@time max_sharpe!(emv)
w3 = copy(emv.weights)
obj3 = objective_value(emv.model)

findmin([obj1, obj2, obj3])

emsv = EffMeanSemivar(tickers, mean_ret, Matrix(returns));
@time max_sharpe!(emsv, optimiser = ECOS.Optimizer)
w1 = copy(emsv.weights)
obj1 = objective_value(emsv.model)

emsv = EffMeanSemivar(tickers, mean_ret, Matrix(returns));
@time max_sharpe!(emsv, optimiser = ECOS.Optimizer)
w2 = copy(emsv.weights)
obj2 = objective_value(emsv.model)

emsv = EffMeanSemivar(tickers, mean_ret, Matrix(returns));
@time max_sharpe!(emsv)
w3 = copy(emsv.weights)
obj3 = objective_value(emsv.model)

findmax([obj1, obj2, obj3])