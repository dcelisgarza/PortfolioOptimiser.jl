using JuMP, Ipopt
using PortfolioOptimiser

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
tickers = names(hist_prices[!, 2:end])

emad = EffMeanAbsDev(tickers, mean_ret, Matrix(returns))
min_risk!(emad, optimiser = Ipopt.Optimizer)
portfolio_performance(emad, verbose = true)

# emad = EffMeanAbsDev(tickers, mean_ret, Matrix(returns))
efficient_return!(emad, optimiser = Ipopt.Optimizer)
portfolio_performance(emad, verbose = true)

# emad = EffMeanAbsDev(tickers, mean_ret, Matrix(returns))
efficient_risk!(emad, 0.5, optimiser = Ipopt.Optimizer)
portfolio_performance(emad, verbose = true)

emad = EffMeanAbsDev(tickers, mean_ret, Matrix(returns), rf = eps())
max_sharpe!(emad, optimiser = Ipopt.Optimizer)
portfolio_performance(emad, verbose = true)

# Try nonlinear optimization
function sharpe_mad_ratio(w::T...) where {T}
    mean_ret = obj_params[1]
    rf = obj_params[2]
    freq = obj_params[3]
    num_tickers = obj_params[4]

    weights = w[1:num_tickers]
    n = w[(num_tickers + 1):end]

    mu = PortfolioOptimiser.port_return(weights, mean_ret) - rf
    sigma = sum(n) / length(n) * freq

    return -mu / sigma
end

nl_emad = EffMeanAbsDev(tickers, mean_ret, Matrix(returns), rf = eps())

obj_params = [nl_emad.mean_ret, nl_emad.rf, nl_emad.freq, length(tickers)]
custom_nloptimiser!(
    nl_emad,
    sharpe_mad_ratio,
    obj_params,
    [(nl_emad.model[:n], 1 / length(nl_emad.model[:n]))],
)

portfolio_performance(nl_emsv, verbose = true)

nl_emad_sharpe_weights = [
    0.01604332649627141,
    0.03295268452523003,
    0.05035385834264111,
    0.017059972873500293,
    0.08269355731801104,
    6.454142572217534e-6,
    0.023580313289956665,
    0.024629982474686115,
    0.013435222400410767,
    0.04280140361944625,
    0.11651487952391605,
    1.898980383402208e-6,
    0.01816668841744695,
    0.05565882234170233,
    0.009419525833495096,
    0.054130767682213816,
    0.09740061162228958,
    0.09606071084350519,
    0.13826946316326774,
    0.11081985610905391,
]

nl_emad_sharpe_perf = (0.09490585178657714, 0.0845039099800302, 1.123094207226682)

emad_sharpe_weights = [
    0.014891817506937908,
    0.03397343162674471,
    0.0492792026652153,
    0.017611047164176788,
    0.08263248964430213,
    5.623880356946985e-9,
    0.02362212139521125,
    0.02407091075523234,
    0.01327215558630656,
    0.04171129647201245,
    0.11927471534789094,
    9.757118016493556e-10,
    0.018488990233931902,
    0.054071740330952876,
    0.009854273322404012,
    0.05371167053312045,
    0.09817146660440665,
    0.0961666888016265,
    0.13832763482356134,
    0.11086834058637372,
]

emad_sharpe_perf = (0.09488945401189575, 0.08422828832879145, 1.1265746448685674)

using ECOS, Ipopt

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