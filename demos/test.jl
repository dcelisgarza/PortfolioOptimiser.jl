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

# mean_ret = ret_model(
#     ECAPMRet(),
#     Matrix(returns),
#     # cspan = num_rows,
#     # rspan = num_rows,
# cov_type = CustomCov(),
# custom_cov_estimator = method,
#     freq = freq,
#     cspan = Int(
#         ceil(
#             size(returns, 1) * log2(min(size(returns, 1), 252)) /
#             log2(max(size(returns, 1), 252)),
#         ),
#     ),
#     rspan = Int(
#         ceil(
#             size(returns, 1) * log2(min(size(returns, 1), 252)) /
#             log2(max(size(returns, 1), 252)),
#         ),
#     ),
#     rf = 1.02^(2048 / 252) - 1,
# )

using PortfolioOptimiser, DataFrames
using Plots, CovarianceEstimation
using LinearAlgebra, JuMP, IJulia
using CSV, MarketData, Statistics, Ipopt, ECOS
using PortfolioOptimiser: max_return
using PortfolioOptimiser: min_risk

hist_prices = CSV.read("./demos/assets/stock_prices.csv", DataFrame)
dropmissing!(hist_prices)

returns = returns_from_prices(hist_prices[!, 2:end])

tickers = names(hist_prices[!, 2:end])

target = DiagonalCommonVariance()
shrinkage = :oas
method = LinearShrinkage(target, shrinkage)

S = cov(CustomCov(), Matrix(returns), estimator = method)
capm_ret = returns_from_prices(
    Matrix(hist_prices[!, 2:end]),
    capm = true,
    cov_type = CustomCov(),
    custom_cov_estimator = method,
)
capm_ret = DataFrame(capm_ret, tickers)

#! EDaR and EVaR https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/
#! NOC https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3572435

mean_ret = ret_model(EMRet(), Matrix(returns))
cmean_ret = ret_model(
    ECAPMRet(),
    Matrix(returns),
    cov_type = CustomCov(),
    custom_cov_estimator = method,
)

emv1 = EffMeanVar(tickers, mean_ret, S)
min_risk!(emv1)
target = sqrt(value(emv1.model[:risk]))

efficient_risk!(emv1, target * 2)
mu, risk, sr = portfolio_performance(emv1, verbose = true)

emv = EffMeanVar(tickers, mean_ret, S)
noc = NearOptCentering(emv)
optimise!(
    noc,
    max_sharpe!,
    # target = 1.03^(1 / 252) - 1,
    # target = target * 2,
    # nlsilent = false,
    # n = 1,
    n = 100,
)

cdarp = EffCDaR(tickers, mean_ret, Matrix(returns), rf = 0)
max_sharpe!(cdarp)
mu, risk = portfolio_performance(cdarp, verbose = true)

################################################################
mean_ret = (1 .+ mean_ret) .^ 252 .- 1
cmean_ret = (1 .+ cmean_ret) .^ 252 .- 1

cdarp = EffCDaR(tickers, mean_ret, Matrix(returns))
noc1 = NearOptCentering(deepcopy(cdarp))
optimise!(noc1, min_risk!, n = 500)

alloc, rem = Allocation(LP(), noc1, Vector(hist_prices[end, tickers]), investment = 650)
alloco, remo =
    Allocation(LP(), noc1.opt_port, Vector(hist_prices[end, tickers]), investment = 650)

munoc1, risknoc1, srnoc1, (mu1, risk1) = portfolio_performance(noc1, verbose = true)

nocs = NearOptCentering(deepcopy(cdarp))
optimise!(nocs, max_sharpe!, n = 500)
munocs, risknocs, srnocs, (mus, risks) = portfolio_performance(nocs, verbose = true)

mrisk = (risks - risk1) / 2
target1 = risk1 + mrisk
target2 = risks + mrisk

noc2 = NearOptCentering(deepcopy(cdarp))
optimise!(noc2, efficient_risk!, target = target1, n = 500)
munoc2, risknoc2, srnoc2, (mu2, risk2) = portfolio_performance(noc2, verbose = true)

noc3 = NearOptCentering(deepcopy(cdarp))
optimise!(noc3, efficient_risk!, target = target2, n = 500)
munoc3, risknoc3, srnoc3, (mu3, risk3) = portfolio_performance(noc3, verbose = true)

display(
    [mean_ret noc1.weights noc2.weights nocs.weights noc3.weights noc1.opt_port.weights noc2.opt_port.weights nocs.opt_port.weights noc3.opt_port.weights],
)

edarp = EffEDaR(tickers, mean_ret, Matrix(returns))
ed1 = deepcopy(edarp)
min_risk!(ed1, optimiser = ECOS.Optimizer)
mu1, risk1 = portfolio_performance(ed1, verbose = true)

eds = deepcopy(edarp)
max_sharpe!(eds, optimiser = ECOS.Optimizer)
mus, risks = portfolio_performance(eds, verbose = true)

mrisk = (risks - risk1) / 2
target1 = risk1 + mrisk
target2 = risks + mrisk

ed2 = deepcopy(edarp)
efficient_risk!(ed2, target1, optimiser = ECOS.Optimizer)
mu2, risk2 = portfolio_performance(ed2, verbose = true)

ed3 = deepcopy(edarp)
efficient_risk!(ed3, target2, optimiser = ECOS.Optimizer)
mu3, risk3 = portfolio_performance(ed3, verbose = true)

display(
    [mean_ret noc1.opt_port.weights noc2.opt_port.weights nocs.opt_port.weights noc3.opt_port.weights ed1.weights eds.weights ed2.weights ed3.weights],
    # ./
    # maximum(
    #     [mean_ret noc1.opt_port.weights noc2.opt_port.weights nocs.opt_port.weights noc3.opt_port.weights ed1.weights eds.weights ed2.weights ed3.weights],
    #     dims = 1,
    # ),
)

#######################

cvarp = EffCVaR(tickers, mean_ret, Matrix(returns))
noc1 = NearOptCentering(deepcopy(cvarp))
optimise!(noc1, min_risk!, n = 500)
munoc1, risknoc1, srnoc1, (mu1, risk1) = portfolio_performance(noc1, verbose = true)

nocs = NearOptCentering(deepcopy(cvarp))
optimise!(nocs, max_sharpe!, n = 500)
munocs, risknocs, srnocs, (mus, risks) = portfolio_performance(nocs, verbose = true)

mrisk = (risks - risk1) / 2
target1 = risk1 + mrisk
target2 = risks + mrisk

noc2 = NearOptCentering(deepcopy(cvarp))
optimise!(noc2, efficient_risk!, target = target1, n = 500)
munoc2, risknoc2, srnoc2, (mu2, risk2) = portfolio_performance(noc2, verbose = true)

noc3 = NearOptCentering(deepcopy(cvarp))
optimise!(noc3, efficient_risk!, target = target2, n = 500)
munoc3, risknoc3, srnoc3, (mu3, risk3) = portfolio_performance(noc3, verbose = true)

display(
    [mean_ret noc1.weights noc2.weights nocs.weights noc3.weights noc1.opt_port.weights noc2.opt_port.weights nocs.opt_port.weights noc3.opt_port.weights],
)

evarp = EffEVaR(tickers, mean_ret, Matrix(returns))
ev1 = deepcopy(evarp)
min_risk!(ev1, optimiser = ECOS.Optimizer)
mu1, risk1 = portfolio_performance(ev1, verbose = true)

evs = deepcopy(evarp)
max_sharpe!(evs, optimiser = ECOS.Optimizer, optimiser_attributes = ("maxit" => 10000))
mus, risks = portfolio_performance(evs, verbose = true)

mrisk = (risks - risk1) / 2
target1 = risk1 + mrisk
target2 = risks + mrisk

ev2 = deepcopy(evarp)
efficient_risk!(ev2, target1, optimiser = ECOS.Optimizer)
mu2, risk2 = portfolio_performance(ev2, verbose = true)

ev3 = deepcopy(evarp)
efficient_risk!(ev3, target2, optimiser = ECOS.Optimizer)
mu3, risk3 = portfolio_performance(ev3, verbose = true)

#######################
w1 = copy(noc.weights)
[value(noc.model[:risk]) / value(noc.opt_port.model[:k]) value(noc.opt_port.model[:risk]) /
                                                         value(noc.opt_port.model[:k])]

cdarp = EffCDaR(tickers, mean_ret, Matrix(returns), rf = 0)
max_sharpe!(cdarp, optimiser = ECOS.Optimizer)
cdmu, cdarval = portfolio_performance(cdarp, verbose = true)

edar = EffEDaR(tickers, mean_ret, Matrix(returns))
min_risk!(edar, optimiser = ECOS.Optimizer, optimiser_attributes = ("maxit" => 250))
mu, edarval = portfolio_performance(edar, verbose = true)

max_utility!(edar, 1, optimiser = ECOS.Optimizer, optimiser_attributes = ("maxit" => 1000))
mu, edarval = portfolio_performance(edar, verbose = true)

efficient_return!(edar, 0.000857976343613329, optimiser = ECOS.Optimizer)
mu, edarval = portfolio_performance(edar, verbose = true)

efficient_risk!(edar, 0.08893937246790279, optimiser = ECOS.Optimizer)
mu, edarval = portfolio_performance(edar, verbose = true)

edar = EffEDaR(tickers, mean_ret, Matrix(capm_ret), rf = 0)
@time max_sharpe!(edar, optimiser = ECOS.Optimizer)
mu, edarval = portfolio_performance(edar, verbose = true)

evar = EffEVaR(tickers, mean_ret, Matrix(returns))
min_risk!(evar, optimiser = ECOS.Optimizer)
mu, evarval = portfolio_performance(evar, verbose = true)

max_utility!(evar, optimiser = ECOS.Optimizer)
mu, evarval = portfolio_performance(evar, verbose = true)

efficient_return!(evar, 0.0013072523137606362, optimiser = ECOS.Optimizer)
mu, evarval = portfolio_performance(evar, verbose = true)

efficient_risk!(evar, 0.03613095055291283, optimiser = ECOS.Optimizer)
mu, evarval = portfolio_performance(evar, verbose = true)

evar = EffEVaR(tickers, mean_ret, Matrix(returns))
@time max_sharpe!(
    evar,
    optimiser = ECOS.Optimizer,
    # optimiser_attributes = ("maxiter" => 1000, "abstol" => 1e-6),
)
mu, evarval = portfolio_performance(evar, verbose = true)

ulcer = EffUlcer(tickers, (1 .+ mean_ret) .^ (252) .- 1, Matrix(returns), rf = 0.02)
min_risk!(ulcer, optimiser = ECOS.Optimizer)
max_utility!(ulcer, optimiser = ECOS.Optimizer)
efficient_return!(ulcer, 0.0009387518959186161, optimiser = ECOS.Optimizer)
efficient_risk!(ulcer, 0.0219 * 3, optimiser = ECOS.Optimizer)

ulcer = EffUlcer(tickers, mean_ret, Matrix(returns))
@time max_sharpe!(ulcer, optimiser = ECOS.Optimizer)

mu, uidx = portfolio_performance(ulcer, verbose = true)

ecvar = EffMeanDaR(tickers, mean_ret, Matrix(returns), rf = 1.02^(1 / 252) - 1)#, extra_obj_terms=[quote L2_reg(model[:w], 2) end])

max_utility!(ecvar, optimiser = Ipopt.Optimizer)
mu, vcvar = portfolio_performance(ecvar, verbose = true)
w = copy(ecvar.weights)

display([ecvar.tickers ecvar.weights])

emimax = EffMinimax(tickers, mean_ret, Matrix(returns), rf = 0)
min_risk!(emimax, optimiser = Ipopt.Optimizer)
portfolio_performance(emimax, verbose = true)

max_utility!(emimax, optimiser = Ipopt.Optimizer)
portfolio_performance(emimax, verbose = true)

emimax = EffMinimax(tickers, mean_ret, Matrix(returns), rf = 0)
max_utility!(emimax)
portfolio_performance(emimax, verbose = true)

emad = EffMeanAbsDev(tickers, mean_ret, Matrix(returns), rf = 0, freq = freq)
min_risk!(emad, optimiser = Ipopt.Optimizer)
portfolio_performance(emad, verbose = true)

emad = EffMeanAbsDev(tickers, mean_ret, Matrix(returns), rf = 0, freq = freq)
max_utility!(emad)
portfolio_performance(emad, verbose = true)

efcvar = EffCVaR(tickers, mean_ret, Matrix(returns), rf = 0)
min_risk!(efcvar)
portfolio_performance(efcvar, verbose = true)

max_utility!(efcvar)
portfolio_performance(efcvar, verbose = true)

efcvar = EffCVaR(tickers, mean_ret, Matrix(returns), rf = 0)
max_sharpe!(efcvar)
portfolio_performance(efcvar, verbose = true)

# emad = EffMeanAbsDev(tickers, mean_ret, Matrix(returns))
efficient_return!(emad, optimiser = Ipopt.Optimizer)
portfolio_performance(emad, verbose = true)

# emad = EffMeanAbsDev(tickers, mean_ret, Matrix(returns))
efficient_risk!(emad, 0.5, optimiser = Ipopt.Optimizer)
portfolio_performance(emad, verbose = true)

emad = EffMeanAbsDev(tickers, mean_ret, Matrix(returns), rf = 0, freq = 252)
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

eminimax = EffMinimax(tickers, mean_ret, Matrix(returns), rf = eps())
min_risk!(eminimax)
portfolio_performance(eminimax, verbose = true)

eminimax.weights

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