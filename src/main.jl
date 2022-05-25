using Statistics, StatsBase, PortfolioOptimiser
using CSV, DataFrames, TimeSeries, LinearAlgebra, JuMP, Ipopt, ECOS

# Reading in the data; preparing expected returns and a risk model
df = CSV.read("./test/assets/stock_prices.csv", DataFrame)
tickers = names(df)[2:end]
returns = returns_from_prices(df[!, 2:end])

mu = vec(ret_model(MRet(), Matrix(dropmissing(returns))))
S = risk_matrix(Cov(), Matrix(dropmissing(returns)))

n = length(names(df)[2:end])
prev_weights = fill(1 / length(names(df)[2:end]), length(names(df)[2:end]))
prev_weights = rand(n)
prev_weights /= sum(prev_weights)

# JuMP doesn't support norm in the objective, so we need to turn them into constraints with MOI.NormOneCone.
# extra_vars = [:(z[1:($n)])]
# extra_constraints =
#     [:([$k * (model[:w] - $prev_weights); model[:z]] in MOI.NormOneCone($(n + n)))]
# Is equivalent to adding k * norm(model[:w] - prev_weights, 1) to the objective
k = 0.001
ef = MeanVar(
    names(df)[2:end],
    mu,
    S;
    extra_vars = [:(z[1:($n)])],
    extra_constraints = [
        :([$k * (model[:w] - $prev_weights); model[:z]] in MOI.NormOneCone($(n + n))),
    ],
    # extra_obj_terms = [quote
    #     L2_reg(model[:w], 0.05)
    # end],
)
min_volatility!(ef)
display(ef.weights)

display(sum([k * ef.model[:w][i] - k * prev_weights[i] for i in 1:length(prev_weights)]))
max_sharpe!(ef)

function mean_barrier_objective(w, cov_matrix, k = 0.1)
    mean_sum = sum(w) / length(w)
    var = dot(w, cov_matrix, w)
    return var - k * mean_sum
end
# function mean_barrier_objective(w::T...) where {T}
#     # quote
#     cov_mtx = obj_params[1]
#     k = obj_params[2]
#     w = [i for i in w]
#     mean_barrier_objective(w, cov_mtx, k)
#     # end
# end
ef = MeanVar(names(df)[2:end], mu, S;)
obj_params = [ef.cov_mtx, 1000000.001]
custom_optimiser!(ef, mean_barrier_objective, obj_params)

function wak(w)
    return dot(w, w)
end
ef = MeanVar(names(df)[2:end], mu, S;)
custom_optimiser!(ef, wak)

# Now try with a non convex objective from  Kolm et al (2014)
function logarithmic_barrier2(w, cov_mtx, k = 0.1)
    # Add eps() to avoid log(0) divergence.
    log_sum = sum(log.(w .+ eps()))
    var = dot(w, cov_mtx, w)
    return var - k * log_sum
end
import PortfolioOptimiser: logarithmic_barrier
function logarithmic_barrier(w::T...) where {T}
    cov_mtx = obj_params[1]
    k = obj_params[2]
    w = [i for i in w]
    PortfolioOptimiser.logarithmic_barrier(w, cov_mtx, k)
    # logarithmic_barrier2(w, cov_mtx, k)
end
ef = MeanVar(names(df)[2:end], mu, S)
obj_params = [ef.cov_mtx, 0.001]
custom_nloptimiser!(ef, logarithmic_barrier, obj_params)

obj_params = (ef.mean_ret, ef.cov_mtx, 1000)
custom_optimiser!(ef, kelly_objective, obj_params)

# Kelly objective with weight bounds on first asset
lower_bounds, upper_bounds = 0.01, 0.3
ef = MeanVar(
    names(df)[2:end],
    mu,
    S;
    extra_constraints = [
        :(model[:w][1] >= $lower_bounds),
        :(model[:w][1] <= $upper_bounds),
    ],
)
obj_params = [ef.mean_ret, ef.cov_mtx, 1000]
custom_optimiser!(ef, kelly_objective, obj_params)

ef = MeanVar(
    names(df)[2:end],
    mu,
    S;
    # extra_obj_terms = [quote
    #     L2_reg(model[:w], 1000)
    # end],
    # extra_obj_terms = [quote
    #     ragnar(model[:w], 1000)
    # end],
    # extra_obj_terms = [quote
    #     function lothar(w, i)
    #         w[5] * w[3] - i
    #     end
    # end],
)

# Now try with a nonconvex objective from  Kolm et al (2014)
function deviation_risk_parity2(w, cov_mtx)
    tmp = w .* (cov_mtx * w) / dot(w, cov_mtx, w)
    return sum((tmp .- 1 / length(w)) .^ 2)
    # return sum(diff .* diff)
end

deviation_risk_parity(ef.weights, ef.cov_mtx)
deviation_risk_parity2(ef.weights, ef.cov_mtx)

@generated function deviation_risk_parity2(w::T...) where {T}
    quote
        cov_mtx = obj_params
        w = [i for i in w]
        deviation_risk_parity(w, cov_mtx)
    end
end

ef = MeanVar(
    names(df)[2:end],
    mu,
    S;
    extra_constraints = [:(model[:w][1] == 0.1)],
    # extra_obj_terms = [quote
    #     L2_reg(model[:w], 1000)
    # end],
    # extra_obj_terms = [quote
    #     ragnar(model[:w], 1000)
    # end],
    # extra_obj_terms = [quote
    #     function lothar(w, i)
    #         w[5] * w[3] - i
    #     end
    # end],
)
obj_params = ef.cov_mtx
custom_nloptimiser!(ef, deviation_risk_parity2, obj_params)

# Now try with a nonconvex objective from  Kolm et al (2014)
function deviation_risk_parity(w, cov_mtx)
    tmp = w .* (cov_mtx * w)
    diff = tmp .- tmp'
    return sum(diff .* diff)
end

@generated function deviation_risk_parity(w::T...) where {T}
    quote
        cov_mtx = obj_params
        w = [i for i in w]
        deviation_risk_parity(w, cov_mtx)
    end
end

obj_params = ef.cov_mtx
custom_nloptimiser!(ef, deviation_risk_parity, obj_params)
testweights = [
    0.051454512685994,
    0.052495693408016,
    0.04672186772311,
    0.042696174644532,
    0.044954259991844,
    0.05883506198745,
    0.029394830327411,
    0.074692721542761,
    0.038158465075084,
    0.048085082859006,
    0.076163515959687,
    0.034353673653056,
    0.027676787623678,
    0.064183101147794,
    0.036666040780056,
    0.043972747264416,
    0.05226758174384,
    0.07059143472995,
    0.045720677583482,
    0.060915769268832,
]
isapprox(ef.weights, testweights, rtol = 1e-1)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.09713251002640265, 0.14588119846495262, 0.5287350997800682
isapprox(mu, mutest, rtol = 1e-2)
isapprox(sigma, sigmatest, rtol = 1e-2)
isapprox(sr, srtest, rtol = 1e-1)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [1, 3, 3, 2, 46, 30, 9, 13, 12, 22, 21, 83, 8, 25, 6, 3, 20, 4, 10]
round.(lpAlloc.shares) == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [21, 8, 20, 8, 10, 45, 3, 3, 1, 12, 3, 4, 6, 3, 13, 24, 20, 29, 83]
rmsd(Int.(round.(gAlloc.shares)), testshares) < 1

###############Alloc.tickers#########
# Black litterman
spy_prices = CSV.read("./test/assets/spy_prices.csv", DataFrame)
delta = market_implied_risk_aversion(spy_prices[!, 2])
deltatest = 2.685491066228311
delta ≈ deltatest

# In the order of the dataframes, the
mcapsdf = DataFrame(
    ticker = names(df)[2:end],
    mcap = [
        927e9,
        1.19e12,
        574e9,
        533e9,
        867e9,
        96e9,
        43e9,
        339e9,
        301e9,
        51e9,
        61e9,
        78e9,
        0,
        295e9,
        1e9,
        22e9,
        288e9,
        212e9,
        422e9,
        102e9,
    ],
)

prior = market_implied_prior_returns(mcapsdf[!, 2], S, delta)
priortest = [
    0.100018238563725
    0.095817277146265
    0.101974120071953
    0.103623740430225
    0.113305470289775
    0.064596564538619
    0.106300826264041
    0.052573811340803
    0.092296370665787
    0.073364357412511
    0.048205218070282
    0.095916237673261
    0.068649071964450
    0.062299939432165
    0.067232761782777
    0.067106130903036
    0.084316356398802
    0.058800474037087
    0.083208859167023
    0.071788977328806
]
prior ≈ priortest

# 1. SBUX drop by 20%
# 2. GOOG outperforms FB by 10%
# 3. BAC and JPM will outperform T and GE by 15%
views = [-0.20, 0.10, 0.15]
picking =
    hcat(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
    )'

bl = BlackLitterman(
    mcapsdf[!, 1],
    S;
    rf = 0,
    tau = 0.01,
    pi = prior, # either a vector, `nothing`, `:equal`, or `:market`
    Q = views,
    P = picking,
)
testpost_ret = [
    0.058404059186963
    0.045896711623189
    0.007453320742957
    0.060355089645467
    0.041326156056687
    -0.000031349301321
    0.077691247532764
    0.011640256362282
    0.091419219703375
    0.030831889341423
    0.003819579599605
    0.012885885501672
    -0.005959314336322
    0.039596385262949
    0.049524436696886
    0.023233278987958
    0.035771855459317
    0.032196366540610
    0.070539152651673
    -0.059373291000260
]
bl.post_ret ≈ testpost_ret
testpost_cov = vcat(
    [0.05334874453292 0.024902672047444 0.035820877126815 0.03003762293178 0.040239757633216 0.015753746209349 0.020901320606043 0.010599290285722 0.022969634599781 0.018184452941796 0.010240781936638 0.025973258402243 0.009064297749123 0.013538208101076 0.006210715239239 0.012688234416409 0.025017676973642 0.01496297568946 0.020016194441763 0.021040600912291],
    [0.024902672047444 0.05378655346829 0.027715737668323 0.024028034151111 0.027240121725497 0.014780532959058 0.033098711831861 0.011552819915695 0.023921951319338 0.019261317981843 0.010624375436169 0.023173090065358 0.018121979504474 0.014968035434157 0.013747654871833 0.018860638211729 0.023508873867951 0.012939357937947 0.02174053531735 0.018137037509882],
    [0.035820877126815 0.027715737668323 0.065315687317338 0.030317625879565 0.038957855421918 0.017024676408514 0.029722501289929 0.00980478073907 0.023954242508466 0.019314377339082 0.009908592917906 0.03499366030552 0.014762971727342 0.013323558842692 0.010996319122858 0.014300277970999 0.025812084646399 0.013774651234565 0.020627276954368 0.022289714337395],
    [0.03003762293178 0.024028034151111 0.030317625879565 0.101918357017587 0.033684859429564 0.016546732063455 0.051026767291063 0.008792539122271 0.024821010516939 0.02070838698998 0.007265433614312 0.028160522674305 0.025507344746638 0.014305718590147 0.024501686107163 0.019770838251757 0.025634708805639 0.013261994393431 0.021428453897999 0.0175104551891],
    [0.040239757633216 0.027240121725497 0.038957855421918 0.033684859429564 0.084186441326594 0.013102039631964 0.030683114990461 0.009329532359921 0.022338707473804 0.016723425860667 0.009469164201705 0.030239737474177 0.004868424300702 0.012996415834839 0.019688863589902 0.014462589996749 0.026523493457822 0.011781491403383 0.019067922153415 0.025290726153949],
    [0.015753746209349 0.014780532959058 0.017024676408514 0.016546732063455 0.013102039631964 0.045546322618926 0.024053819130086 0.009721114482515 0.025348180325503 0.022066907386658 0.013407954548335 0.027118500000968 0.029271861523519 0.017666148809851 0.025616703559727 0.01211576330532 0.016440745985032 0.014279107809269 0.023222340135006 0.015556744457971],
    [0.020901320606043 0.033098711831861 0.029722501289929 0.051026767291063 0.030683114990461 0.024053819130086 0.419435740109041 0.011477565295854 0.042579570395332 0.035219728639183 0.016410203945159 0.026858724945831 0.035576381829818 0.02279674386249 0.061254442712256 0.030485215262191 0.033182344528116 0.018843469973971 0.031989975975234 0.013556548739957],
    [0.010599290285722 0.011552819915695 0.00980478073907 0.008792539122271 0.009329532359921 0.009721114482515 0.011477565295854 0.041603952007734 0.010508522499114 0.012654353044667 0.010775500238374 0.016264694466671 0.02839435328056 0.008767439301597 0.005296035825415 0.014670575655231 0.011681472878172 0.011308790475922 0.011995716652398 0.012350793595876],
    [0.022969634599781 0.023921951319338 0.023954242508466 0.024821010516939 0.022338707473804 0.025348180325503 0.042579570395332 0.010508522499114 0.070059650164303 0.031932681675489 0.011914877668708 0.035953252498342 0.037689078480604 0.023206827652217 0.03438071083178 0.030032807360198 0.027858949081679 0.01968588510039 0.051021491390749 0.019709534765143],
    [0.018184452941796 0.019261317981843 0.019314377339082 0.02070838698998 0.016723425860667 0.022066907386658 0.035219728639183 0.012654353044667 0.031932681675489 0.061876666057302 0.012119544562166 0.032548730414819 0.037031741557304 0.016678576740828 0.026459306692323 0.025405843850092 0.01951833438857 0.015293548042868 0.027455167376016 0.017700258458998],
    [0.010240781936638 0.010624375436169 0.009908592917906 0.007265433614312 0.009469164201705 0.013407954548335 0.016410203945159 0.010775500238374 0.011914877668708 0.012119544562166 0.026795081644734 0.012420626767153 0.014970292581857 0.012347504980603 0.015922535723888 0.010251092702776 0.010843846272749 0.009480106941065 0.01309306276484 0.00914149735524],
    [0.025973258402243 0.023173090065358 0.03499366030552 0.028160522674305 0.030239737474177 0.027118500000968 0.026858724945831 0.016264694466671 0.035953252498342 0.032548730414819 0.012420626767153 0.179749910435694 0.058493054783132 0.019223964528325 0.033173828688863 0.034920875598853 0.023589018847215 0.017337377334031 0.030907137690852 0.027376522902746],
    [0.009064297749123 0.018121979504474 0.014762971727342 0.025507344746638 0.004868424300702 0.029271861523519 0.035576381829818 0.02839435328056 0.037689078480604 0.037031741557304 0.014970292581857 0.058493054783132 0.494697541906723 0.019898343296291 0.056376935787249 0.059452838044951 0.015111514351748 0.014180557030523 0.025439788469831 0.023979222437031],
    [0.013538208101076 0.014968035434157 0.013323558842692 0.014305718590147 0.012996415834839 0.017666148809851 0.02279674386249 0.008767439301597 0.023206827652217 0.016678576740828 0.012347504980603 0.019223964528325 0.019898343296291 0.036539546031524 0.039313250146948 0.012002339660102 0.016367538036895 0.013024216890089 0.022285558956566 0.010168475613478],
    [0.006210715239239 0.013747654871833 0.010996319122858 0.024501686107163 0.019688863589902 0.025616703559727 0.061254442712256 0.005296035825415 0.03438071083178 0.026459306692323 0.015922535723888 0.033173828688863 0.056376935787249 0.039313250146948 0.252892111331342 0.029103381607604 0.01702654624531 0.00955948287177 0.0295607068289 0.008895620469753],
    [0.012688234416409 0.018860638211729 0.014300277970999 0.019770838251757 0.014462589996749 0.01211576330532 0.030485215262191 0.014670575655231 0.030032807360198 0.025405843850092 0.010251092702776 0.034920875598853 0.059452838044951 0.012002339660102 0.029103381607604 0.128579631489709 0.018970727664979 0.012512528618383 0.02346950763889 0.019336816446419],
    [0.025017676973642 0.023508873867951 0.025812084646399 0.025634708805639 0.026523493457822 0.016440745985032 0.033182344528116 0.011681472878172 0.027858949081679 0.01951833438857 0.010843846272749 0.023589018847215 0.015111514351748 0.016367538036895 0.01702654624531 0.018970727664979 0.041281485121884 0.015560942620389 0.024957639540371 0.019587608239884],
    [0.01496297568946 0.012939357937947 0.013774651234565 0.013261994393431 0.011781491403383 0.014279107809269 0.018843469973971 0.011308790475922 0.01968588510039 0.015293548042868 0.009480106941065 0.017337377334031 0.014180557030523 0.013024216890089 0.00955948287177 0.012512528618383 0.015560942620389 0.031037346425707 0.018369433119826 0.011288056098657],
    [0.020016194441763 0.02174053531735 0.020627276954368 0.021428453897999 0.019067922153415 0.023222340135006 0.031989975975234 0.011995716652398 0.051021491390749 0.027455167376016 0.01309306276484 0.030907137690852 0.025439788469831 0.022285558956566 0.0295607068289 0.02346950763889 0.024957639540371 0.018369433119826 0.046919986197744 0.01794461920481],
    [0.021040600912291 0.018137037509882 0.022289714337395 0.0175104551891 0.025290726153949 0.015556744457971 0.013556548739957 0.012350793595876 0.019709534765143 0.017700258458998 0.00914149735524 0.027376522902746 0.023979222437031 0.010168475613478 0.008895620469753 0.019336816446419 0.019587608239884 0.011288056098657 0.01794461920481 0.039986416856178],
)
bl.post_cov ≈ testpost_cov

testblweights = [
    2.802634574339719,
    1.0696569899116537,
    -1.5225894339745898,
    0.5221972605398294,
    0.7399073537664403,
    -1.3989450429613053,
    0.00932451280100568,
    0.6505588964261255,
    1.8362707796803164,
    0.09084468616612303,
    -0.8183433897385306,
    0.0075712114558542905,
    -0.024199601308436784,
    0.6270484617832653,
    0.002865041743474888,
    0.08532234626520806,
    0.27339562941444895,
    0.7122779313389186,
    1.7936495956277718,
    -6.459447803277293,
]
bl.weights ≈ testblweights

## Efficient Frontier
ef = MeanVar(names(df)[2:end], bl.post_ret, S)
max_sharpe!(ef)

import PortfolioOptimiser: max_sharpe_nl!, L2_reg

function L2_reg(γ = 1, w...)
    return γ * dot(w, w)
end

ef2 = MeanVar(names(df)[2:end], bl.post_ret, S, extra_obj_terms = [quote
    L2_reg(1000, w...)
end])

extra_obj_terms = [quote
    L2_reg(1000, w...)
end]
max_sharpe_nl!(ef2)

testweights = [
    0.221896132231067,
    0.0,
    0.0,
    0.067801981190784,
    0.0,
    0.0,
    0.018545141283091,
    0.0,
    0.691756745295058,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
isapprox(ef.weights, testweights, rtol = 1e-5)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.08173248653446699, 0.2193583713684134, 0.2814229798906876
isapprox(mu, mutest, rtol = 1e-5)
isapprox(sigma, sigmatest, rtol = 1e-5)
isapprox(sr, srtest, rtol = 1e-5)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [2, 4, 32, 232]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [231, 2, 4, 19]
gAlloc.shares == testshares

ef = MeanVar(names(df)[2:end], bl.post_ret, S)
min_volatility!(ef)
testweights = [
    0.007909381852655,
    0.030690045397095,
    0.010506892855973,
    0.027486977795442,
    0.012277615067487,
    0.033411624151502,
    0.0,
    0.139848395652807,
    0.0,
    0.0,
    0.287822361245511,
    0.0,
    0.0,
    0.125283674542888,
    0.0,
    0.015085474081069,
    0.0,
    0.193123876049454,
    0.0,
    0.116553681308117,
]
isapprox(ef.weights, testweights, rtol = 1e-4)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.011450350358250433, 0.1223065907437908, -0.06990342539805944
isapprox(mu, mutest, rtol = 1e-4)
isapprox(sigma, sigmatest, rtol = 1e-6)
isapprox(sr, srtest, rtol = 1e-3)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [2, 1, 2, 26, 16, 82, 16, 2, 54, 20]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [81, 54, 16, 16, 20, 25, 2, 2, 3, 1]
gAlloc.shares == testshares

max_quadratic_utility!(ef)
testweights = [
    0.142307545055626,
    0.0,
    0.0,
    0.045165241857315,
    0.0,
    0.0,
    0.019010547882617,
    0.0,
    0.793516665204442,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
isapprox(ef.weights, testweights, rtol = 1e-5)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.0850569180412342, 0.2324517001148201, 0.27987284244038296
isapprox(mu, mutest, rtol = 1e-6)
isapprox(sigma, sigmatest, rtol = 1e-6)
isapprox(sr, srtest, rtol = 1e-6)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [1, 3, 20, 276]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [265, 1, 3, 19]
gAlloc.shares == testshares

efficient_risk!(ef, 0.2)
testweights = [
    2.682965379536906e-01,
    1.301727613248030e-02,
    6.941516000000000e-10,
    8.082980723587720e-02,
    2.778368900000000e-09,
    7.370853000000000e-10,
    1.569566834242780e-02,
    3.096453200000000e-09,
    5.290628313405414e-01,
    1.599880500000000e-09,
    1.590630700000000e-09,
    6.629340000000000e-10,
    5.250114000000000e-10,
    6.508987253961621e-02,
    4.401409430000000e-08,
    1.435398500000000e-09,
    2.166394700000000e-09,
    1.627557970000000e-08,
    2.800793056420930e-02,
    3.151541000000000e-10,
]
isapprox(ef.weights, testweights, rtol = 1e-4)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.07528445695449039, 0.1999999998533396, 0.2764222849751529
isapprox(mu, mutest, rtol = 1e-8)
isapprox(sigma, sigmatest, rtol = 1e-6)
isapprox(sr, srtest, rtol = 1e-6)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [3, 4, 13, 176, 8, 2]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [176, 2, 5, 9, 3, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1]
gAlloc.shares == testshares

efficient_return!(ef, 0.07)
testweights = [
    0.248314255432587,
    0.034008459762881,
    0.0,
    0.07614301178335,
    0.0,
    0.0,
    0.012512720842415,
    0.0,
    0.425848064696512,
    0.0,
    0.0,
    0.0,
    0.0,
    0.11002301819797,
    0.0,
    0.0,
    0.0,
    0.038837837899579,
    0.054312631384705,
    0.0,
]
isapprox(ef.weights, testweights, rtol = 1e-4)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.07, 0.18628889663990178, 0.26840032284184107
isapprox(mu, mutest, rtol = 1e-6)
isapprox(sigma, sigmatest, rtol = 1e-7)
isapprox(sr, srtest, rtol = 1e-6)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [2, 2, 5, 13, 143, 15, 11, 5]
rmsd(Int.(lpAlloc.shares), testshares) <= 1

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [142, 2, 14, 5, 5, 11, 2, 13]
gAlloc.shares == testshares

### Eff front market neutral
ef = MeanVar(
    names(df)[2:end],
    bl.post_ret,
    S,
    market_neutral = true,
    weight_bounds = (-1, 1),
)
max_sharpe!(ef)
testweights = [
    0.762986154593938,
    0.296385852834282,
    -0.486716965453677,
    0.177003190230186,
    0.121993477778985,
    -0.463311778342318,
    0.012676176673992,
    0.190543821955541,
    0.55195291200316,
    -0.004864099330881,
    -0.105903955137662,
    -0.039570283245285,
    -0.024098048248515,
    0.304351666002147,
    0.01578308228419,
    -0.001253599586287,
    -0.038726173479159,
    0.31185831103011,
    0.418910257437251,
    -1.0,
]
isapprox(ef.weights, testweights, rtol = 1e-7)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.23339921135959843, 0.3472734444465163, 0.6144990778080186
isapprox(mu, mutest, rtol = 1e-7)
isapprox(sigma, sigmatest, rtol = 1e-7)
isapprox(sr, srtest, rtol = 1e-9)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [3, 5, 3, 2, 7, 58, 12, 3, 25, 12, -29, -358, -2, -30, -24, -74, -2, -169]
rmsd(Int.(lpAlloc.shares), testshares) <= 1

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [3, 58, 11, 27, 12, 5, 7, 3, 4, 4, -168, -29, -357, -30, -24, -2, -73, -2, -1]
gAlloc.shares == testshares

ef = MeanVar(
    names(df)[2:end],
    bl.post_ret,
    S,
    market_neutral = true,
    weight_bounds = (-1, 1),
)
min_volatility!(ef)
testweights = [
    0.003589375488559,
    0.03755470565059,
    0.017678840134291,
    0.033086142381968,
    0.01248643676087,
    0.053795551664664,
    -0.009710932107096,
    0.141218424731254,
    -0.010896109687934,
    0.018593267935107,
    0.283682357130765,
    -0.021141221569855,
    -0.009132893914897,
    0.145883284989944,
    0.000773588362254,
    0.025431696052056,
    0.014568945811608,
    0.20358571785576,
    -0.064209856959109,
    0.123162679289201,
]
isapprox(ef.weights, testweights, rtol = 1e-6)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.007547970562401391, 0.12111738873121844, -0.10280959297456398
isapprox(mu, mutest, rtol = 1e-6)
isapprox(sigma, sigmatest, rtol = 1e-14)
isapprox(sr, srtest, rtol = 1e-6)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [2, 1, 2, 37, 15, 4, 72, 17, 1, 3, 1, 51, 19, -10, -3, -12, -28, -6]
rmsd(Int.(lpAlloc.shares), testshares) <= 0.5

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [72, 51, 17, 15, 19, 37, 2, 2, 3, 4, 1, 1, 1, -6, -12, -3, -10, -28]
gAlloc.shares == testshares

max_quadratic_utility!(ef)
testweights = [
    1.0,
    0.426913849108477,
    -0.899112930395218,
    0.276568855383093,
    0.135625407796987,
    -1.0,
    0.055969477291634,
    -0.033008319585208,
    1.0,
    -0.090464327162872,
    -0.770684036606438,
    -0.067011238393225,
    -0.04140724550793,
    0.310653348488697,
    0.033237184469854,
    -0.104174950464479,
    -0.215315619639611,
    0.169206567068369,
    0.813003978147871,
    -1.0,
]
isapprox(ef.weights, testweights, rtol = 1e-5)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.30865288246835887, 0.492856336854342, 0.585673472944849
isapprox(mu, mutest, rtol = 1e-5)
isapprox(sigma, sigmatest, rtol = 1e-5)
isapprox(sr, srtest, rtol = 1e-6)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares =
    [2, 6, 4, 44, 81, 11, 6, 12, 18, -54, -771, -4, -23, -219, -40, -126, -15, -12, -169]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares =
    [3, 79, 17, 5, 9, 3, 11, 14, 5, -771, -168, -54, -218, -13, -14, -23, -40, -126, -4]
gAlloc.shares == testshares

efficient_risk!(ef, 0.2)
testweights = [
    4.011943951916352e-01,
    1.479341183051060e-01,
    -2.207710538722699e-01,
    7.010589737916791e-02,
    1.042631570686915e-01,
    -2.082259922788151e-01,
    2.728431826645200e-03,
    7.300470985808140e-02,
    2.647605713529499e-01,
    1.035586817410430e-02,
    -1.579565672582347e-01,
    4.115326096184400e-03,
    -2.159594074949800e-03,
    6.896697187466180e-02,
    2.997857007137000e-04,
    8.584124489862901e-03,
    3.709806063010500e-02,
    7.291225417342059e-02,
    2.662917880916170e-01,
    -9.435022527286794e-01,
]
isapprox(ef.weights, testweights, rtol = 1e-5)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.1435925344169654, 0.19999999989333053, 0.6179626724144157
isapprox(mu, mutest, rtol = 1e-6)
isapprox(sigma, sigmatest, rtol = 1e-6)
isapprox(sr, srtest, rtol = 1e-7)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [3, 6, 3, 2, 6, 58, 2, 2, 6, 1, 1, 1, 14, 16, -13, -162, -45, -7, -159]
rmsd(lpAlloc.shares, testshares) <= 0.5

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [2, 15, 57, 6, 1, 5, 13, 3, 6, 1, 1, 1, 2, -158, -13, -160, -45, -7]
gAlloc.shares == testshares

efficient_return!(ef, 0.07)
testweights = [
    1.955786393513531e-01,
    7.211643474355210e-02,
    -1.076236901818929e-01,
    3.417582789309590e-02,
    5.082732970587820e-02,
    -1.015078387959005e-01,
    1.330070131094100e-03,
    3.558932040881150e-02,
    1.290677217451837e-01,
    5.048447975984000e-03,
    -7.700222068272380e-02,
    2.006233620643200e-03,
    -1.052761181828400e-03,
    3.362061842342390e-02,
    1.461368350657000e-04,
    4.184759818585200e-03,
    1.808508509531880e-02,
    3.554402444179850e-02,
    1.298148468599680e-01,
    -4.599489862074102e-01,
]
isapprox(ef.weights, testweights, rtol = 1e-4)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.06999999999999999, 0.09749810496318828, 0.5128304803347528
isapprox(mu, mutest, rtol = 1e-8)
isapprox(sigma, sigmatest, rtol = 1e-8)
isapprox(sr, srtest, rtol = 1e-8)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [3, 6, 3, 2, 6, 58, 2, 2, 6, 1, 1, 1, 14, 16, -6, -79, -22, -4, -78]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [2, 15, 57, 6, 1, 5, 13, 3, 6, 1, 1, 1, 2, -77, -6, -78, -22, -4]
gAlloc.shares == testshares

## Mean port_semivar
ef = MeanSemivar(tickers, bl.post_ret, Matrix(dropmissing(returns)), benchmark = 0)
max_sortino!(ef)
mumax, varmax, smax = portfolio_performance(ef)

ef = MeanSemivar(tickers, bl.post_ret, Matrix(dropmissing(returns)), benchmark = 0)
efficient_risk!(ef, varmax)
mu, sigma, sr = portfolio_performance(ef)
isapprox(mu, mumax, rtol = 1e-5)
isapprox(sigma, varmax, rtol = 1e-5)
isapprox(sr, smax, rtol = 1e-6)

efficient_return!(ef, mumax)
mu, sigma, sr = portfolio_performance(ef)
isapprox(mu, mumax, rtol = 1e-5)
isapprox(sigma, varmax, rtol = 1e-4)
isapprox(sr, smax, rtol = 1e-4)

ef = MeanSemivar(tickers, bl.post_ret, Matrix(dropmissing(returns)), benchmark = 0)
min_semivar!(ef)

testweights = [
    -8.836959900000000e-09,
    5.211306673060680e-02,
    -5.546965560000000e-08,
    3.273135667801300e-03,
    3.246505644507460e-02,
    6.374472200000000e-09,
    3.371255680000000e-07,
    1.255659243075475e-01,
    1.633273845000000e-07,
    4.589898500000000e-08,
    3.061337299353124e-01,
    -6.684586460000000e-08,
    -1.418747400000000e-08,
    9.674675422017130e-02,
    4.653018274916000e-03,
    1.988382240191780e-02,
    1.809257784940300e-03,
    2.315968397352792e-01,
    1.032235721000000e-07,
    1.257588839643445e-01,
]
isapprox(ef.weights, testweights, rtol = 1e-1)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.011139799284510227, 0.08497381732464267, -0.1042697738485651
isapprox(mu, mutest, atol = 1e-2)
isapprox(sigma, sigmatest, rtol = 1e-3)
isapprox(sr, srtest, atol = 1e-1)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [3, 1, 1, 2, 15, 87, 13, 4, 3, 65, 22]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [86, 64, 21, 15, 13, 3, 3, 3, 1, 1, 1]
gAlloc.shares == testshares

max_quadratic_utility!(ef)
testweights = [
    1.100000000000000e-15,
    1.200000000000000e-15,
    1.100000000000000e-15,
    1.000000000000000e-15,
    1.200000000000000e-15,
    1.000000000000000e-15,
    5.000000000000000e-16,
    1.500000000000000e-15,
    9.999999999999784e-01,
    9.000000000000000e-16,
    1.500000000000000e-15,
    8.000000000000000e-16,
    7.000000000000000e-16,
    1.200000000000000e-15,
    9.000000000000000e-16,
    1.100000000000000e-15,
    1.000000000000000e-15,
    1.300000000000000e-15,
    5.000000000000000e-16,
    1.300000000000000e-15,
]
isapprox(ef.weights, testweights, rtol = 1e-3)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.091419219703374, 0.1829724238719523, 0.39032777831786586
isapprox(mu, mutest, rtol = 1e-3)
isapprox(sigma, sigmatest, rtol = 1e-3)
isapprox(sr, srtest, rtol = 1e-4)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [334]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [334, 1]
gAlloc.shares == testshares

efficient_risk!(ef, 0.13)
testweights = [
    2.874750375415112e-01,
    6.195275902493500e-02,
    4.152863104000000e-07,
    3.754883121164440e-02,
    2.335468505800000e-06,
    4.139595347000000e-07,
    5.180351073718700e-03,
    2.282885531500000e-06,
    4.040036784043188e-01,
    8.569500938000000e-07,
    1.073950326300000e-06,
    4.148599173000000e-07,
    3.056287327000000e-07,
    5.362159168381390e-02,
    3.496258672140000e-05,
    1.238539893500000e-06,
    1.317666202400000e-06,
    6.146402530123640e-02,
    8.870790761308990e-02,
    2.002169627000000e-07,
]
isapprox(ef.weights, testweights, rtol = 1e-2)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.06959704494304779, 0.1300556974853854, 0.38135234289617415
isapprox(mu, mutest, rtol = 1e-3)
isapprox(sigma, sigmatest, rtol = 1e-3)
isapprox(sr, srtest, rtol = 1e-3)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [3, 3, 2, 135, 7, 17, 8]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [135, 3, 8, 3, 17, 6, 2, 6, 1, 1]
gAlloc.shares == testshares

efficient_return!(ef, 0.09)
testweights = [
    3.904746815385050e-02,
    5.923397958900000e-06,
    9.829962419900001e-06,
    4.485521037400000e-06,
    6.398473463000000e-06,
    1.058079920190000e-05,
    8.609310912354300e-03,
    9.421584972200000e-06,
    9.522038516156038e-01,
    7.426588225800000e-06,
    1.019932942990000e-05,
    9.315528047900000e-06,
    1.118486159000000e-05,
    6.553049888500000e-06,
    5.498591175600000e-06,
    8.239081503600000e-06,
    6.940805469200000e-06,
    7.319651581700000e-06,
    3.430088268500000e-06,
    1.661143469400000e-05,
]
isapprox(ef.weights, testweights, rtol = 1e-2)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.09000090563810152, 0.17811334827609804, 0.393013248673487
isapprox(mu, mutest, rtol = 1e-5)
isapprox(sigma, sigmatest, rtol = 1e-4)
isapprox(sr, srtest, rtol = 1e-4)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [1, 9, 319, 1, 1, 1, 1, 1, 1, 1, 1, 1]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [318, 9, 1, 1, 1, 1, 1, 1, 1, 1]
gAlloc.shares == testshares

### Mean port_semivar market neutral
ef = MeanSemivar(
    tickers,
    bl.post_ret,
    Matrix(dropmissing(returns)),
    benchmark = 0,
    market_neutral = true,
    weight_bounds = (-1, 1),
)
min_semivar!(ef)
testweights = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
isapprox(ef.weights, testweights, atol = 1e-1)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0, 0, 0
isapprox(mu, mutest, atol = 1e-2)
isapprox(sigma, sigmatest, atol = 1e-2)

ef.weights .= testweights
lpAlloc, remaining =
    Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000, silent = true)
testshares = nothing
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = nothing
gAlloc.shares == testshares

max_quadratic_utility!(ef)
testweights = [
    0.9999999987191048,
    0.8025061567310201,
    -0.9999999983660633,
    0.6315188600428658,
    0.4922749743437155,
    -0.9999999990605686,
    0.1762382499326327,
    -0.733252470272586,
    0.9999999993131146,
    -0.2585591997727487,
    -0.9999999983706372,
    -0.2591719865317014,
    -0.1281535712856913,
    0.4788874233441203,
    -0.0263096469774795,
    -0.0874576034104335,
    -0.4552360118252822,
    0.3667148242725146,
    0.9999999989095066,
    -0.9999999997354032,
]
isapprox(ef.weights, testweights, rtol = 1e-2)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.3695444374167076, 0.4300511641888641, 0.8127973286062292
isapprox(mu, mutest, rtol = 1e-3)
isapprox(sigma, sigmatest, rtol = 1e-3)
isapprox(sr, srtest, rtol = 1e-3)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [
    1,
    8,
    6,
    1,
    30,
    57,
    11,
    17,
    15,
    -60,
    -771,
    -85,
    -66,
    -284,
    -155,
    -387,
    -17,
    -12,
    -27,
    -168,
]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [
    57,
    15,
    1,
    8,
    6,
    1,
    11,
    17,
    30,
    -168,
    -771,
    -284,
    -60,
    -85,
    -27,
    -154,
    -66,
    -389,
    -12,
    -17,
]
gAlloc.shares == testshares

efficient_risk!(ef, 0.13)
testweights = [
    0.423840175531549,
    0.153868620456201,
    -0.17273587531483,
    0.075346138406724,
    0.14756120611187,
    -0.264240220855537,
    0.011500806938995,
    0.036277198349417,
    0.164956472823101,
    -0.011353011722617,
    -0.140741456448818,
    0.021451677598097,
    -0.001559330611481,
    0.016650555284113,
    -0.010666436683385,
    0.02763414585217,
    -0.019506524583918,
    0.121695189218685,
    0.369932229128064,
    -0.949911559486228,
]
isapprox(ef.weights, testweights, rtol = 1e-3)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.14346415111692887, 0.1300415287657269, 0.9494209448994764
isapprox(mu, mutest, rtol = 1e-3)
isapprox(sigma, sigmatest, rtol = 1e-3)
isapprox(sr, srtest, rtol = 1e-4)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [2, 6, 3, 1, 7, 3, 35, 7, 2, 3, 22, 21, -10, -204, -3, -40, -8, -11, -1, -160]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [2, 21, 35, 6, 1, 21, 3, 3, 3, 8, 2, 8, -159, -204, -11, -39, -1, -3, -7, -4]
rmsd(Int.(gAlloc.shares), testshares) < 4

efficient_return!(ef, 0.09)
testweights = [
    2.666155000843805e-01,
    1.000205615502960e-01,
    -1.109461468510381e-01,
    4.458166169899530e-02,
    9.324272424737901e-02,
    -1.650002620767060e-01,
    7.865259980471200e-03,
    2.291134032560510e-02,
    1.051873711703434e-01,
    -9.433352247161599e-03,
    -9.115863996696000e-02,
    1.332391558924420e-02,
    4.224537248974000e-04,
    1.161933418260970e-02,
    -6.157199603787500e-03,
    1.670878511393480e-02,
    -1.312104640268140e-02,
    7.464390852176560e-02,
    2.321656069881314e-01,
    -5.934917762008577e-01,
]
isapprox(ef.weights, testweights, rtol = 5e-2)
mu, sigma, sr = portfolio_performance(ef)
mutest, sigmatest, srtest = 0.09000000684283793, 0.08158778822807418, 0.8579716200561383
isapprox(mu, mutest, rtol = 1e-4)
isapprox(sigma, sigmatest, rtol = 1e-3)
isapprox(sr, srtest, rtol = 1e-3)

ef.weights .= testweights
lpAlloc, remaining = Allocation(LP(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [2, 6, 3, 1, 8, 3, 35, 8, 2, 2, 3, 21, 21, -7, -126, -2, -26, -3, -1, -99]
rmsd(lpAlloc.shares, testshares) < 1

gAlloc, remaining =
    Allocation(Greedy(), ef, Vector(df[end, ef.tickers]); investment = 10000)
testshares = [2, 21, 35, 6, 1, 21, 3, 3, 3, 8, 2, 8, 2, -99, -127, -7, -25, -1, -2, -4]
rmsd(Int.(gAlloc.shares), testshares) < 3

## CVaR
cv = EfficientCVaR(tickers, bl.post_ret, Matrix(dropmissing(returns)))
min_cvar!(cv, ECOS.Optimizer)
testweights = [
    1.196700000000000e-12,
    4.242033345496910e-02,
    1.247300000000000e-12,
    3.092200000000000e-12,
    7.574027808866600e-03,
    1.305400000000000e-12,
    1.735000000000000e-13,
    9.464947870552300e-02,
    5.519000000000000e-13,
    7.337000000000000e-13,
    3.040110655916818e-01,
    6.947000000000000e-13,
    3.879000000000000e-13,
    6.564167130410940e-02,
    1.189600000000000e-12,
    2.937161110264830e-02,
    1.548900000000000e-12,
    3.663101128889494e-01,
    7.108000000000000e-13,
    9.002169913041960e-02,
]
isapprox(cv.weights, testweights, rtol = 1e-3)
mu, sigma = portfolio_performance(cv)
mutest, sigmatest = 0.014253439792781208, 0.017049502122532846
isapprox(mu, mutest, rtol = 1e-3)
isapprox(sigma, sigmatest, rtol = 1e-3)

cv.weights .= testweights
lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [3, 11, 86, 8, 4, 103, 15]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [102, 86, 11, 16, 8, 3, 4]
gAlloc.shares == testshares

efficient_return!(cv, 0.07, ECOS.Optimizer)
testweights = [
    3.839859145444188e-01,
    3.596981096365410e-02,
    6.780000000000000e-14,
    7.786600000000000e-12,
    6.220000000000000e-13,
    9.530000000000000e-14,
    8.145000000000000e-13,
    8.463000000000000e-13,
    4.466960681039454e-01,
    2.409000000000000e-13,
    3.971000000000000e-13,
    1.291000000000000e-13,
    4.190000000000000e-14,
    1.071394799383709e-01,
    6.634200000000000e-12,
    4.764000000000000e-13,
    3.925000000000000e-13,
    2.620872642514560e-02,
    5.954000000000000e-12,
    -3.320000000000000e-14,
]
isapprox(cv.weights, testweights, rtol = 1e-3)
mu, sigma = portfolio_performance(cv)
mutest, sigmatest = 0.06999999999999391, 0.028255572821056847
isapprox(mu, mutest, rtol = 1e-7)
isapprox(sigma, sigmatest, rtol = 1e-4)

cv.weights .= testweights
lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [4, 2, 147, 12, 7]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [149, 3, 13, 4, 7, 3, 1]
gAlloc.shares == testshares

efficient_risk!(cv, 0.14813)
testweights = [
    2.850000000000000e-14,
    1.580000000000000e-14,
    3.500000000000000e-15,
    -5.300000000000000e-15,
    2.060000000000000e-14,
    1.300000000000000e-15,
    3.591000000000000e-13,
    9.200000000000000e-15,
    9.999999999994234e-01,
    1.310000000000000e-14,
    6.900000000000000e-15,
    0.000000000000000e+00,
    -5.400000000000000e-15,
    1.530000000000000e-14,
    8.599999999999999e-15,
    9.600000000000000e-15,
    1.800000000000000e-14,
    1.950000000000000e-14,
    6.730000000000000e-14,
    -9.400000000000000e-15,
]
isapprox(cv.weights, testweights, atol = 1e-5)
mu, sigma = portfolio_performance(cv)
mutest, sigmatest = 0.09141921970336199, 0.148136422458718
isapprox(mu, mutest, rtol = 1e-3)
isapprox(sigma, sigmatest, rtol = 1e-3)

cv.weights .= testweights
lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [334]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [334, 1]
gAlloc.shares == testshares

### Market neutral
cv = EfficientCVaR(
    tickers,
    bl.post_ret,
    Matrix(dropmissing(returns)),
    market_neutral = true,
    weight_bounds = (-1, 1),
)
min_cvar!(cv)
testweights = [
    -2.2800e-12,
    -5.5010e-13,
    5.6220e-13,
    -3.1620e-13,
    5.1780e-12,
    -4.6641e-12,
    8.1920e-13,
    -1.2637e-12,
    -2.9427e-12,
    2.0640e-13,
    3.8680e-13,
    -1.8054e-12,
    -5.7220e-13,
    -3.0342e-12,
    -1.6556e-12,
    1.6296e-12,
    2.5777e-12,
    -4.7750e-13,
    8.4177e-12,
    -2.1610e-13,
]
isapprox(cv.weights, testweights, atol = 1e-4)
mu, sigma = portfolio_performance(cv)
mutest, sigmatest = 3.279006735579045e-13, 5.19535240389637e-13
isapprox(mu, mutest, atol = 1e-1)
isapprox(sigma, sigmatest, atol = 1e-1)

cv.weights .= testweights
lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [2, 2, 41, 2, 4, 11, 7, 38]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [38, 2, 7, 12, 43, 1, 5, 3]
rmsd(Int.(gAlloc.shares), testshares) < 2

efficient_return!(cv, 0.07)
testweights = [
    0.119904777207769,
    0.105260613527609,
    -0.059118345889513,
    0.047994829008171,
    0.080535064283592,
    -0.07038968929214,
    -0.000695786742806,
    0.054006019393697,
    0.088529728836899,
    -0.01637772952951,
    -0.063653950435258,
    0.012986548220159,
    0.019165300733247,
    0.021582076971202,
    -0.00296786688154,
    0.018096675589053,
    -0.024160465458667,
    0.072693909074444,
    0.137412159501253,
    -0.540803868117662,
]
isapprox(cv.weights, testweights, rtol = 1e-3)
mu, sigma = portfolio_performance(cv)
mutest, sigmatest = 0.07000000000000023, 0.01208285385909055
isapprox(mu, mutest, rtol = 1e-7)
isapprox(sigma, sigmatest, rtol = 1e-3)

cv.weights .= testweights
lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [1, 8, 4, 1, 8, 38, 10, 75, 4, 3, 26, 16, -4, -54, -1, -4, -18, -2, -1, -91]
rmsd(lpAlloc.shares, testshares) < 0.5

gAlloc, remaining =
    Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [16, 1, 8, 38, 1, 26, 8, 4, 4, 75, 3, 10, -91, -54, -18, -4, -1, -4, -2, -1]
rmsd(Int.(gAlloc.shares), testshares) < 1

efficient_risk!(cv, 0.18)
testweights = [
    0.999999999998078,
    0.999999999994987,
    -0.999999999998141,
    0.999999999996761,
    0.999999999951026,
    -0.9999999999986,
    0.999999999934693,
    -0.999999999995853,
    0.999999999999143,
    -0.999999998594101,
    -0.999999999998166,
    -0.999999999986599,
    -0.83525041495262,
    0.999999999989784,
    0.709272218478147,
    -0.163990750353505,
    -0.345299781283349,
    -0.364731273180876,
    0.999999999998686,
    -0.999999999999495,
]
isapprox(cv.weights, testweights, rtol = 1e-4)
mu, sigma = portfolio_performance(cv)
mutest, sigmatest = 0.4902004788284181, 0.1799999999999532
isapprox(mu, mutest, rtol = 1e-5)
isapprox(sigma, sigmatest, rtol = 1e-5)

cv.weights .= testweights
lpAlloc, remaining = Allocation(LP(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [
    1,
    7,
    6,
    1,
    117,
    38,
    14,
    54,
    10,
    -60,
    -771,
    -117,
    -256,
    -284,
    -597,
    -2533,
    -23,
    -20,
    -102,
    -168,
]
rmsd(lpAlloc.shares, testshares) < 0.5

gAlloc, remaining =
    Allocation(Greedy(), cv, Vector(df[end, cv.tickers]); investment = 10000)
testshares = [
    38,
    10,
    1,
    6,
    7,
    14,
    1,
    117,
    54,
    -168,
    -771,
    -284,
    -60,
    -117,
    -597,
    -256,
    -2531,
    -102,
    -20,
    -23,
]
gAlloc.shares == testshares

## Efficient CDaR
cdar = EfficientCDaR(tickers, bl.post_ret, Matrix(dropmissing(returns)))
min_cdar!(cdar)
testweights = [
    5.500000000000000e-15,
    4.210000000000000e-13,
    1.010000000000000e-14,
    4.000000000000000e-16,
    3.409901158173600e-03,
    -3.300000000000000e-15,
    -1.070000000000000e-14,
    7.904282392519640e-02,
    1.250000000000000e-14,
    -2.900000000000000e-15,
    3.875931701996790e-01,
    4.740000000000000e-14,
    6.370000000000000e-14,
    1.063000000000000e-13,
    5.545152701592000e-04,
    9.598828930536250e-02,
    2.679088825496902e-01,
    1.995000000000000e-13,
    6.560171960161000e-04,
    1.648464003948736e-01,
]
isapprox(cdar.weights, testweights, rtol = 1e-3)
mu, sigma = portfolio_performance(cdar)
mutest, sigmatest = 0.0046414239877397775, 0.05643312227060557
isapprox(mu, mutest, rtol = 1e-2)
isapprox(sigma, sigmatest, rtol = 1e-4)

cdar.weights .= testweights
lpAlloc, remaining =
    Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares = [9, 110, 13, 16, 28]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares = [109, 16, 27, 14, 9, 1, 1]
gAlloc.shares == testshares

efficient_return!(cdar, 0.071)
testweights = [
    1.803135468327820e-01,
    -7.000000000000000e-16,
    -1.100000000000000e-15,
    -8.000000000000000e-16,
    -9.000000000000000e-16,
    -1.200000000000000e-15,
    3.736518284742810e-02,
    -8.000000000000000e-16,
    1.446130925597511e-01,
    -1.100000000000000e-15,
    -1.000000000000000e-15,
    -1.100000000000000e-15,
    -1.200000000000000e-15,
    5.000000000000000e-16,
    3.035022274167050e-02,
    -9.000000000000000e-16,
    -9.000000000000000e-16,
    -9.000000000000000e-16,
    6.073579550183820e-01,
    -1.300000000000000e-15,
]
isapprox(cdar.weights, testweights, rtol = 1e-4)
mu, sigma = portfolio_performance(cdar)
mutest, sigmatest = 0.07099999999999985, 0.14924652616273293
isapprox(mu, mutest, rtol = 1e-6)
isapprox(sigma, sigmatest, rtol = 1e-5)

cdar.weights .= testweights
lpAlloc, remaining =
    Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares = [2, 27, 48, 19, 54]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares = [54, 1, 48, 38, 20, 1]
gAlloc.shares == testshares

efficient_risk!(cdar, 0.11)
testweights = [
    3.411720847078938e-01,
    4.200000000000000e-15,
    6.000000000000000e-16,
    -4.000000000000000e-16,
    3.050000000000000e-14,
    -7.000000000000000e-16,
    1.300000000000000e-14,
    3.767304612523230e-02,
    6.100000000000000e-15,
    -2.000000000000000e-16,
    2.916000000000000e-13,
    6.210000000000000e-14,
    2.300000000000000e-15,
    6.480796061687440e-02,
    1.261914308485920e-02,
    3.719518575826500e-02,
    1.120000000000000e-14,
    6.700000000000000e-15,
    5.065325797064497e-01,
    -1.400000000000000e-15,
]
isapprox(cdar.weights, testweights, rtol = 1e-3)
mu, sigma = portfolio_performance(cdar)
mutest, sigmatest = 0.060150020563327425, 0.11000000000000373
isapprox(mu, mutest, rtol = 1e-4)
isapprox(sigma, sigmatest, rtol = 1e-7)

cdar.weights .= testweights
lpAlloc, remaining =
    Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares = [3, 6, 9, 10, 6, 46]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares = [45, 3, 8, 10, 5, 8]
gAlloc.shares == testshares

### Market neutral
cdar = EfficientCDaR(
    tickers,
    bl.post_ret,
    Matrix(dropmissing(returns)),
    weight_bounds = (-1, 1),
    market_neutral = true,
)
min_cdar!(cdar, ECOS.Optimizer)
testweights = [
    -7.8660e-13,
    -5.1690e-13,
    5.5350e-13,
    2.5550e-13,
    1.0952e-12,
    -1.6305e-12,
    2.7030e-13,
    -2.7810e-13,
    -1.7863e-12,
    -1.7250e-13,
    3.4360e-13,
    -4.9670e-13,
    -1.1400e-13,
    -6.5110e-13,
    -2.9790e-13,
    5.5730e-13,
    4.9150e-13,
    -6.2820e-13,
    2.9461e-12,
    8.4570e-13,
]
isapprox(cdar.weights, testweights, rtol = 1e-4)
mu, sigma = portfolio_performance(cdar)
mutest, sigmatest = -3.270061253855585e-14, 2.788899603525084e-13
isapprox(mu, mutest, rtol = 1e-3)
isapprox(sigma, sigmatest, rtol = 1e-14)

cdar.weights .= testweights
lpAlloc, remaining =
    Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares = [5, 2, 1, 35, 13, 11, 4, 36, 19]
lpAlloc.shares == testshares

gAlloc, remaining =
    Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares = [36, 1, 19, 10, 5, 4, 14, 38, 2]
rmsd(Int.(gAlloc.shares), testshares) < 0.5

efficient_return!(cdar, 0.071)
testweights = [
    0.025107592782914,
    0.057594547532327,
    -0.029106830395298,
    -0.055379879544535,
    0.277094131770879,
    -0.218125545473876,
    0.01760029260404,
    0.033400679475767,
    -0.361271598572642,
    -0.088479023848706,
    -0.184621283291016,
    -0.023896668602229,
    -0.006314246903803,
    0.277651628420218,
    -0.069684404622742,
    0.084480397324026,
    0.11757646255151,
    -0.008078698917481,
    0.690348263568902,
    -0.535895815858256,
]
isapprox(cdar.weights, testweights, rtol = 1e-4)
mu, sigma = portfolio_performance(cdar)
mutest, sigmatest = 0.07099999999999981, 0.06860632727537126
isapprox(mu, mutest, rtol = 1e-6)
isapprox(sigma, sigmatest, rtol = 1e-4)

cdar.weights .= testweights
lpAlloc, remaining =
    Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares =
    [3, 1, 12, 3, 23, 8, 5, 40, -2, -3, -168, -121, -23, -52, -14, -19, -47, -2, -90]
rmsd(lpAlloc.shares, testshares) < 1

gAlloc, remaining =
    Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares =
    [39, 23, 1, 5, 8, 3, 3, 12, -90, -121, -168, -52, -23, -47, -3, -2, -14, -2, -19]
rmsd(Int.(gAlloc.shares), testshares) < 1

efficient_risk!(cdar, 0.11, ECOS.Optimizer)
testweights = [
    0.028783758447304,
    0.154930722945264,
    -0.052888584214774,
    -0.090606438318005,
    0.438460257717241,
    -0.34165630482699,
    0.02387460049364,
    0.054720588895777,
    -0.519576218315878,
    -0.132186114282055,
    -0.232431229341235,
    -0.006075698448925,
    -0.022163078215555,
    0.434087117994023,
    -0.130403026894326,
    0.14155104241332,
    0.175044671976498,
    -0.036231017136812,
    0.999999999999363,
    -0.887235050887876,
]
isapprox(cdar.weights, testweights, rtol = 1e-14)
mu, sigma = portfolio_performance(cdar)
mutest, sigmatest = 0.11347395924861056, 0.11000000000006227
isapprox(mu, mutest, rtol = 1e-15)
isapprox(sigma, sigmatest, rtol = 1e-15)

cdar.weights .= testweights
lpAlloc, remaining =
    Allocation(LP(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares =
    [4, 1, 10, 3, 23, 9, 5, 37, -3, -5, -263, -174, -34, -66, -4, -68, -87, -10, -150]
rmsd(lpAlloc.shares, testshares) < 0.5

gAlloc, remaining =
    Allocation(Greedy(), cdar, Vector(df[end, cdar.tickers]); investment = 10000)
testshares =
    [36, 1, 22, 4, 4, 8, 3, 10, -149, -173, -263, -66, -33, -86, -5, -4, -10, -67, -3]
rmsd(Int.(gAlloc.shares), testshares) < 2

## HROpt
hrp = HRPOpt(tickers, returns = Matrix(dropmissing(returns)))
idx = sortperm(tickers)
testweights = [
    0.05141029982354615,
    0.012973422626800228,
    0.02018216653246635,
    0.04084621303656177,
    0.015567906456662952,
    0.0377521927556203,
    0.04075799338381744,
    0.06072154753518565,
    0.04354241996221517,
    0.03182464218785058,
    0.02325487286365021,
    0.04956897986914887,
    0.10700323656585976,
    0.017325239748703498,
    0.08269670342726206,
    0.010999466705659471,
    0.15533136214701582,
    0.02353673037019126,
    0.10170965737619252,
    0.07299494662558993,
]
hrp.weights[idx] ≈ testweights
mu, sigma, sr = portfolio_performance(hrp)
mutest, sigmatest, srtest = 0.10779113291906073, 0.1321728564045751, 0.6642145392571078
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest

using PortfolioOptimiser, DataFrames, CSV, Statistics

df = CSV.read("./test/assets/stock_prices.csv", DataFrame)
dropmissing!(df)
returns = returns_from_prices(df[!, 2:end])
cov_mtx = risk_matrix(
    SCov(),
    Matrix(returns),
    1.02^(1 / 252) - 1,
    SFix(),
    252,
    Int(ceil(252 / 1.4)),
)
testcov = reshape(
    [
        0.022525453904467,
        0.014702519870182,
        0.019053982917046,
        0.018696916900512,
        0.018409284534755,
        0.012529128830922,
        0.025945495387098,
        0.00906252641551,
        0.016002832942479,
        0.014837626429073,
        0.008289326315192,
        0.016565859447689,
        0.020363864553773,
        0.0106456062799,
        0.018202853128116,
        0.013186329325946,
        0.014352437901067,
        0.010168834926307,
        0.013642351807908,
        0.012663735203131,
        0.014702519870182,
        0.024877719319234,
        0.016870717350807,
        0.017722225672765,
        0.015812096439494,
        0.011678614245765,
        0.027343097346851,
        0.008513232865114,
        0.015709056305636,
        0.014241616539979,
        0.007973271889648,
        0.017286173384212,
        0.023361014259085,
        0.010741372923038,
        0.020726861755274,
        0.014769673119315,
        0.013711441290225,
        0.00833181521624,
        0.013006276915931,
        0.01114284463531,
        0.019053982917046,
        0.016870717350807,
        0.029504687179255,
        0.022036183656905,
        0.020499385981739,
        0.013801364374561,
        0.029751079162577,
        0.008985313368779,
        0.017402321317274,
        0.016112974285734,
        0.00854913841589,
        0.019936495988079,
        0.024214251349298,
        0.009980421638581,
        0.019849836881366,
        0.015303103580172,
        0.015967297259296,
        0.010850821740143,
        0.01418955073874,
        0.014044230051494,
        0.018696916900512,
        0.017722225672765,
        0.022036183656905,
        0.04492421976902,
        0.021583535546375,
        0.015968076421437,
        0.040335092738185,
        0.010597967619348,
        0.020717405376339,
        0.019522089125673,
        0.010356447924047,
        0.022982694239327,
        0.033494500806262,
        0.013259809899829,
        0.031519029969913,
        0.017840237818766,
        0.01711297092672,
        0.011471372004329,
        0.017110588888627,
        0.014039178633428,
        0.018409284534755,
        0.015812096439494,
        0.020499385981739,
        0.021583535546375,
        0.030720052288252,
        0.012473036680992,
        0.030353854237477,
        0.009351268305371,
        0.017476523182104,
        0.015297753200559,
        0.00823423753673,
        0.019132257038425,
        0.023504172143148,
        0.010334316949157,
        0.022182175733993,
        0.014285336585942,
        0.015802398362235,
        0.009753449058108,
        0.014165665150165,
        0.014562406771882,
        0.012529128830922,
        0.011678614245765,
        0.013801364374561,
        0.015968076421437,
        0.012473036680992,
        0.02440228436048,
        0.024270353503385,
        0.008929498787609,
        0.016590855691834,
        0.015246302436768,
        0.009357190994096,
        0.020838468107746,
        0.031461352497691,
        0.0111529534843,
        0.022433204353138,
        0.012770419417478,
        0.011607514518822,
        0.009310312741978,
        0.013864426966984,
        0.010271579915927,
        0.025945495387098,
        0.027343097346851,
        0.029751079162577,
        0.040335092738185,
        0.030353854237477,
        0.024270353503385,
        0.155039128410935,
        0.015774747228493,
        0.03520296177244,
        0.032434455241135,
        0.018291992172521,
        0.036086319478679,
        0.061043316460493,
        0.020962191067428,
        0.052738849401468,
        0.032510076463945,
        0.026615333851302,
        0.018577521292829,
        0.027791112301525,
        0.02004136380838,
        0.00906252641551,
        0.008513232865114,
        0.008985313368779,
        0.010597967619348,
        0.009351268305371,
        0.008929498787609,
        0.015774747228493,
        0.021040043228967,
        0.010153058123292,
        0.00994413171462,
        0.007118337336687,
        0.013703951963209,
        0.025712628094029,
        0.007965166258197,
        0.013724736058423,
        0.013552473863293,
        0.007872045391554,
        0.007553860717186,
        0.009533706229943,
        0.008250553261039,
        0.016002832942479,
        0.015709056305636,
        0.017402321317274,
        0.020717405376339,
        0.017476523182104,
        0.016590855691834,
        0.03520296177244,
        0.010153058123292,
        0.033703059151725,
        0.01999679985416,
        0.009601367101102,
        0.022904778615797,
        0.032114247249851,
        0.014805757364219,
        0.028375007677103,
        0.018982645552864,
        0.016853429564492,
        0.011901752277136,
        0.024550299661909,
        0.012844360207056,
        0.014837626429073,
        0.014241616539979,
        0.016112974285734,
        0.019522089125673,
        0.015297753200559,
        0.015246302436768,
        0.032434455241135,
        0.00994413171462,
        0.01999679985416,
        0.030013916039327,
        0.010146921400224,
        0.022977718154282,
        0.034760579005305,
        0.012746949467944,
        0.026693425701261,
        0.018036547351302,
        0.013293845137808,
        0.010717987931216,
        0.016439844471517,
        0.012170455458605,
        0.008289326315192,
        0.007973271889648,
        0.00854913841589,
        0.010356447924047,
        0.00823423753673,
        0.009357190994096,
        0.018291992172521,
        0.007118337336687,
        0.009601367101102,
        0.010146921400224,
        0.013322768453092,
        0.011365383230523,
        0.019233470534308,
        0.007930364671191,
        0.015464001707986,
        0.008828942689305,
        0.007422085347014,
        0.006545773475848,
        0.008396896606228,
        0.006707225996955,
        0.016565859447689,
        0.017286173384212,
        0.019936495988079,
        0.022982694239327,
        0.019132257038425,
        0.020838468107746,
        0.036086319478679,
        0.013703951963209,
        0.022904778615797,
        0.022977718154282,
        0.011365383230523,
        0.098323500223113,
        0.049266391473751,
        0.014767159777128,
        0.036819645459222,
        0.025548150829406,
        0.016805891212801,
        0.013044893686802,
        0.018942284744052,
        0.016948695207603,
        0.020363864553773,
        0.023361014259085,
        0.024214251349298,
        0.033494500806262,
        0.023504172143148,
        0.031461352497691,
        0.061043316460493,
        0.025712628094029,
        0.032114247249851,
        0.034760579005305,
        0.019233470534308,
        0.049266391473751,
        0.217250051013063,
        0.022891842887802,
        0.0712810230896,
        0.042122318236915,
        0.020267249870451,
        0.017713825770789,
        0.023354170633112,
        0.022831361630414,
        0.0106456062799,
        0.010741372923038,
        0.009980421638581,
        0.013259809899829,
        0.010334316949157,
        0.0111529534843,
        0.020962191067428,
        0.007965166258197,
        0.014805757364219,
        0.012746949467944,
        0.007930364671191,
        0.014767159777128,
        0.022891842887802,
        0.018268949158542,
        0.025453850018682,
        0.011543173328894,
        0.01044740392348,
        0.008305838702664,
        0.012562806535381,
        0.008419459935667,
        0.018202853128116,
        0.020726861755274,
        0.019849836881366,
        0.031519029969913,
        0.022182175733993,
        0.022433204353138,
        0.052738849401468,
        0.013724736058423,
        0.028375007677103,
        0.026693425701261,
        0.015464001707986,
        0.036819645459222,
        0.0712810230896,
        0.025453850018682,
        0.12537329786879,
        0.02861913492648,
        0.018888305321645,
        0.014190840162381,
        0.022219019674271,
        0.016300305338172,
        0.013186329325946,
        0.014769673119315,
        0.015303103580172,
        0.017840237818766,
        0.014285336585942,
        0.012770419417478,
        0.032510076463945,
        0.013552473863293,
        0.018982645552864,
        0.018036547351302,
        0.008828942689305,
        0.025548150829406,
        0.042122318236915,
        0.011543173328894,
        0.02861913492648,
        0.053688934136254,
        0.012850578683661,
        0.010285493010373,
        0.014886458492273,
        0.012393441822333,
        0.014352437901067,
        0.013711441290225,
        0.015967297259296,
        0.01711297092672,
        0.015802398362235,
        0.011607514518822,
        0.026615333851302,
        0.007872045391554,
        0.016853429564492,
        0.013293845137808,
        0.007422085347014,
        0.016805891212801,
        0.020267249870451,
        0.01044740392348,
        0.018888305321645,
        0.012850578683661,
        0.018888968545482,
        0.009215066292241,
        0.013890907862614,
        0.011030687476662,
        0.010168834926307,
        0.00833181521624,
        0.010850821740143,
        0.011471372004329,
        0.009753449058108,
        0.009310312741978,
        0.018577521292829,
        0.007553860717186,
        0.011901752277136,
        0.010717987931216,
        0.006545773475848,
        0.013044893686802,
        0.017713825770789,
        0.008305838702664,
        0.014190840162381,
        0.010285493010373,
        0.009215066292241,
        0.013588093093942,
        0.01013169888453,
        0.008114476181133,
        0.013642351807908,
        0.013006276915931,
        0.01418955073874,
        0.017110588888627,
        0.014165665150165,
        0.013864426966984,
        0.027791112301525,
        0.009533706229943,
        0.024550299661909,
        0.016439844471517,
        0.008396896606228,
        0.018942284744052,
        0.023354170633112,
        0.012562806535381,
        0.022219019674271,
        0.014886458492273,
        0.013890907862614,
        0.01013169888453,
        0.021570979370674,
        0.010773216311524,
        0.012663735203131,
        0.01114284463531,
        0.014044230051494,
        0.014039178633428,
        0.014562406771882,
        0.010271579915927,
        0.02004136380838,
        0.008250553261039,
        0.012844360207056,
        0.012170455458605,
        0.006707225996955,
        0.016948695207603,
        0.022831361630414,
        0.008419459935667,
        0.016300305338172,
        0.012393441822333,
        0.011030687476662,
        0.008114476181133,
        0.010773216311524,
        0.020237813558209,
    ],
    20,
    :,
)
isapprox(cov_mtx, testcov, rtol = 1e-4)

cov_mtx = risk_matrix(
    ESCov(),
    Matrix(returns),
    1.02^(1 / 252) - 1,
    SFix(),
    252,
    Int(ceil(252 / 1.4)),
)
testcov = reshape(
    [0.03900084445226888 0.02374258088471931 0.035869417622279645 0.03640742649057418 0.028186004966870952 0.027240440548595438 0.04026661312611741 0.0191731384957373 0.02818455631006372 0.028000113283300914 0.017626714412831493 0.022437046504583197 0.031121564573307763 0.02014972137405705 0.03241396792252521 0.02079315932751833 0.024927854004288732 0.018751295520966653 0.025262612711423604 0.014426759595958854 0.02374258088471931 0.02327494227017931 0.022956880172756204 0.024941914379434854 0.019355183380401382 0.01887419883223524 0.03122690192576201 0.012470777830288101 0.01841442909919217 0.018520585572330918 0.01293080903897608 0.018295297099522895 0.03238238463082841 0.014583339503145139 0.027782792063716557 0.016713464235307462 0.017796258078622344 0.011804652050756679 0.016098371263525268 0.010406188430738846 0.035869417622279645 0.022956880172756204 0.05005312642447694 0.0385167550521469 0.03021106687053133 0.029343832605535695 0.04220061285648465 0.021117500926600936 0.02613027586944575 0.029006297542945834 0.01731813629947294 0.023279271685409657 0.0357935896522795 0.017975195294675946 0.03942532201802745 0.021209973453588334 0.025016518054020373 0.018500147232217625 0.022847270389948843 0.015246051405915478 0.03640742649057418 0.024941914379434854 0.0385167550521469 0.05661794399499479 0.035624665781281026 0.0313532851022564 0.051160549485834914 0.01998286255916682 0.028665450585703245 0.02996257532967651 0.019328514288973822 0.03041497497369548 0.03913850317111943 0.019807350849157714 0.044333428438182616 0.023465250862139717 0.027524327210544368 0.018010882985740766 0.026134963257091443 0.01636912812780202 0.028186004966870952 0.019355183380401382 0.03021106687053133 0.035624665781281026 0.03905409950660545 0.022414759478299223 0.041018877367648206 0.01805296347373808 0.02462064241601972 0.020531501385830174 0.013590244712487107 0.019517538178452556 0.03049713660263585 0.013965603213361679 0.03392745726308473 0.016641500616100705 0.02422498639647871 0.0152281956638418 0.020344188452962424 0.015597624891021749 0.027240440548595438 0.01887419883223524 0.029343832605535695 0.0313532851022564 0.022414759478299223 0.06116878239154157 0.04145860144242304 0.022159497319487687 0.026640485466983343 0.025884923514425253 0.021219401548214303 0.03901433850134988 0.06968465612676489 0.020564660411202005 0.04391838454223453 0.022738368113496225 0.019796529648499962 0.019318188309949625 0.02425324727007459 0.014754165842570409 0.04026661312611741 0.03122690192576201 0.04220061285648465 0.051160549485834914 0.041018877367648206 0.04145860144242304 0.1286675794180281 0.028464815708040776 0.03762634439964831 0.03899364092979199 0.029321663551167148 0.04531682206046382 0.07705739272036252 0.02785530576924673 0.06243910933412938 0.03229192142345256 0.033695061673749895 0.02617043957917601 0.032943424226975056 0.01949257551110657 0.0191731384957373 0.012470777830288101 0.021117500926600936 0.01998286255916682 0.01805296347373808 0.022159497319487687 0.028464815708040776 0.04030024827873196 0.017451513041391822 0.019083874824329827 0.014909353660265579 0.019905466812838148 0.05250231813167481 0.016264059254567913 0.026826667619076355 0.020964991158943954 0.015578936943864922 0.015830596436245955 0.015647685114910603 0.01184098370382105 0.02818455631006372 0.01841442909919217 0.02613027586944575 0.028665450585703245 0.02462064241601972 0.026640485466983343 0.03762634439964831 0.017451513041391822 0.033276816662583544 0.02436868793147879 0.017897852589770336 0.023271174776337316 0.035361987289644146 0.016319552560780434 0.028633624519339734 0.01997441607572644 0.022090970319827128 0.018928129761107745 0.027458957840935938 0.012942719383273033 0.028000113283300914 0.018520585572330918 0.029006297542945834 0.02996257532967651 0.020531501385830174 0.025884923514425253 0.03899364092979199 0.019083874824329827 0.02436868793147879 0.035151120365633115 0.01892418295485163 0.027202989304215396 0.04161280681503957 0.01761802214509021 0.035637124370474296 0.020285835550748678 0.019071300050307882 0.01720100975249153 0.022283279935779637 0.012602146173577373 0.017626714412831493 0.01293080903897608 0.01731813629947294 0.019328514288973822 0.013590244712487107 0.021219401548214303 0.029321663551167148 0.014909353660265579 0.017897852589770336 0.01892418295485163 0.022963931568992648 0.01803312268427431 0.032309010925483585 0.014126572282556905 0.02371198004027108 0.016296010422792606 0.013944725368054694 0.013375620084307994 0.016562070185158297 0.009214356456215251 0.022437046504583197 0.018295297099522895 0.023279271685409657 0.03041497497369548 0.019517538178452556 0.03901433850134988 0.04531682206046382 0.019905466812838148 0.023271174776337316 0.027202989304215396 0.01803312268427431 0.13279231919878043 0.06947900898459157 0.020449895817267912 0.04773069173273337 0.029009348773535257 0.016534131366384444 0.017510693061620033 0.022571412515189015 0.014167554885994752 0.031121564573307763 0.03238238463082841 0.0357935896522795 0.03913850317111943 0.03049713660263585 0.06968465612676489 0.07705739272036252 0.05250231813167481 0.035361987289644146 0.04161280681503957 0.032309010925483585 0.06947900898459157 0.29401151056041896 0.030432377210891298 0.0931883397659145 0.05158256719247313 0.024902758676110186 0.028750244403287613 0.030840782861172318 0.0315898977595021 0.02014972137405705 0.014583339503145139 0.017975195294675946 0.019807350849157714 0.013965603213361679 0.020564660411202005 0.02785530576924673 0.016264059254567913 0.016319552560780434 0.01761802214509021 0.014126572282556905 0.020449895817267912 0.030432377210891298 0.02186711936186108 0.02477835186436577 0.017204173676101633 0.013487061890850824 0.013294173072279897 0.016285987597671987 0.009224145577920985 0.03241396792252521 0.027782792063716557 0.03942532201802745 0.044333428438182616 0.03392745726308473 0.04391838454223453 0.06243910933412938 0.026826667619076355 0.028633624519339734 0.035637124370474296 0.02371198004027108 0.04773069173273337 0.0931883397659145 0.02477835186436577 0.11551312455151863 0.03255140938504359 0.02399876199245975 0.02068065927712369 0.025605068984379458 0.020851150065248056 0.02079315932751833 0.016713464235307462 0.021209973453588334 0.023465250862139717 0.016641500616100705 0.022738368113496225 0.03229192142345256 0.020964991158943954 0.01997441607572644 0.020285835550748678 0.016296010422792606 0.029009348773535257 0.05158256719247313 0.017204173676101633 0.03255140938504359 0.050421423681374526 0.016864111093478896 0.015204835869136991 0.01847173671569178 0.010402043967278458 0.024927854004288732 0.017796258078622344 0.025016518054020373 0.027524327210544368 0.02422498639647871 0.019796529648499962 0.033695061673749895 0.015578936943864922 0.022090970319827128 0.019071300050307882 0.013944725368054694 0.016534131366384444 0.024902758676110186 0.013487061890850824 0.02399876199245975 0.016864111093478896 0.024695716660106816 0.01521978322787175 0.019452391679403075 0.01071761421458703 0.018751295520966653 0.011804652050756679 0.018500147232217625 0.018010882985740766 0.0152281956638418 0.019318188309949625 0.02617043957917601 0.015830596436245955 0.018928129761107745 0.01720100975249153 0.013375620084307994 0.017510693061620033 0.028750244403287613 0.013294173072279897 0.02068065927712369 0.015204835869136991 0.01521978322787175 0.01931203418106597 0.017022633262181338 0.00948134108482751 0.025262612711423604 0.016098371263525268 0.022847270389948843 0.026134963257091443 0.020344188452962424 0.02425324727007459 0.032943424226975056 0.015647685114910603 0.027458957840935938 0.022283279935779637 0.016562070185158297 0.022571412515189015 0.030840782861172318 0.016285987597671987 0.025605068984379458 0.01847173671569178 0.019452391679403075 0.017022633262181338 0.025122040099215915 0.010951651986171332 0.014426759595958854 0.010406188430738846 0.015246051405915478 0.01636912812780202 0.015597624891021749 0.014754165842570409 0.01949257551110657 0.01184098370382105 0.012942719383273033 0.012602146173577373 0.009214356456215251 0.014167554885994752 0.0315898977595021 0.009224145577920985 0.020851150065248056 0.010402043967278458 0.01071761421458703 0.00948134108482751 0.010951651986171332 0.01946608891959149],
    20,
    :,
)
testcov ≈ cov_mtx

corr = cov2cor(cov_mtx)
testcorr = reshape(
    [1.0 0.7880370103378278 0.8118423738397118 0.7747755888771443 0.7222095128793865 0.5577148442008776 0.5684254724707756 0.4836178736651935 0.7823540050325126 0.7562287727997502 0.5889945020062183 0.3117756583012488 0.29063121009801907 0.6899796025512063 0.4829248204751438 0.46889534799901644 0.8032247282972168 0.683250978585417 0.8070748362260224 0.5235913198639335 0.7880370103378278 1.0 0.6725939497132537 0.6870817775935776 0.6419772970798221 0.5002179842733188 0.57062399192518 0.4071883792308268 0.6616724704905059 0.6475015639050216 0.559317322386344 0.3290854509219013 0.39145545914565044 0.6464231368641967 0.5358162944590256 0.48788180072620496 0.7422899354678594 0.5567944190945463 0.665748475282032 0.4888864809355612 0.8118423738397118 0.6725939497132537 1.0 0.7235308271375829 0.6833090034450882 0.530317933753682 0.5258581241056541 0.4701896404838719 0.6402616698999543 0.6915237605062178 0.5108135706585807 0.2855402307396717 0.2950580605439317 0.5433281730380624 0.5184942908702888 0.4221987742397929 0.7115418362708285 0.5950391891353299 0.6443046480828952 0.4884301040402025 0.7747755888771443 0.6870817775935776 0.7235308271375829 1.0 0.7576008218966399 0.5327716249403196 0.5994094529735697 0.41833749664270453 0.6604057387346127 0.6716337410455928 0.536041363792235 0.350771207897804 0.30335069785035823 0.562928918146957 0.548199461971616 0.4391780258991724 0.7360864919712798 0.5446831579286184 0.6929747243297233 0.4930706471445408 0.7222095128793865 0.6419772970798221 0.6833090034450882 0.7576008218966399 1.0 0.4586019013506314 0.5786499107807346 0.4550523152613802 0.6829598599794166 0.5541378413970237 0.4538066343257688 0.2710224365720845 0.28460569368296407 0.477892920371326 0.5051290344693149 0.37501757322082124 0.7800445569351906 0.5544994546859855 0.6495006703990495 0.5656995145637783 0.5577148442008776 0.5002179842733188 0.530317933753682 0.5327716249403196 0.4586019013506314 1.0 0.4673208569072034 0.4463146326825705 0.5904815466394996 0.5582289845856291 0.5661677030527795 0.4328853137283263 0.5196250469484923 0.5622906173252058 0.5224753114035746 0.40943675181474465 0.5093465044074202 0.5620661960809631 0.6186963081159329 0.42757311673914256 0.5684254724707756 0.57062399192518 0.5258581241056541 0.5994094529735697 0.5786499107807346 0.4673208569072034 1.0 0.39529375214746487 0.5750247978210401 0.5798152542190169 0.539424811844449 0.3466876755222492 0.39618458911698745 0.5251429952595528 0.5121609570322493 0.4009145053464104 0.5977517735470183 0.5250037933290096 0.5794376638957516 0.3894891001523857 0.4836178736651935 0.4071883792308268 0.4701896404838719 0.41833749664270453 0.4550523152613802 0.4463146326825705 0.39529375214746487 1.0 0.4765498512105587 0.5070410398551982 0.490096762633423 0.27210218430665745 0.4823279749355802 0.5478722443132951 0.3931852360313254 0.46508600438409436 0.49382507887417354 0.567452359469588 0.4917776473106598 0.42276066378519034 0.7823540050325126 0.6616724704905059 0.6402616698999543 0.6604057387346127 0.6829598599794166 0.5904815466394996 0.5750247978210401 0.4765498512105587 1.0 0.7125111639857031 0.6474510176448908 0.350074875463064 0.35750622645086655 0.6049807731583691 0.4618381161019484 0.48763587982473167 0.7706072459010869 0.7466598640213862 0.9496984103271222 0.5085284990098963 0.7562287727997502 0.6475015639050216 0.6915237605062178 0.6716337410455928 0.5541378413970237 0.5582289845856291 0.5798152542190169 0.5070410398551982 0.7125111639857031 1.0 0.6660769887573716 0.3981626946696996 0.40933158552360205 0.6354652082204137 0.5592646909728172 0.4818543262464657 0.6472914883513033 0.6601921339331994 0.7498632642238646 0.4817653659558023 0.5889945020062183 0.559317322386344 0.5108135706585807 0.536041363792235 0.4538066343257688 0.5661677030527795 0.539424811844449 0.490096762633423 0.6474510176448908 0.6660769887573716 1.0 0.32655875931912626 0.3932044049638235 0.630402441122833 0.46039334050493097 0.47890637662513413 0.585566235326388 0.6351507160237317 0.689547274888211 0.4358153628530099 0.3117756583012488 0.3290854509219013 0.2855402307396717 0.350771207897804 0.2710224365720845 0.4328853137283263 0.3466876755222492 0.27210218430665745 0.350074875463064 0.3981626946696996 0.32655875931912626 1.0 0.351629230794864 0.3794974255986497 0.3853856901845631 0.3545226194665939 0.2887247999104022 0.3457823701818523 0.3907912290635113 0.27865643358002923 0.29063121009801907 0.39145545914565044 0.2950580605439317 0.30335069785035823 0.28460569368296407 0.5196250469484923 0.39618458911698745 0.4823279749355802 0.35750622645086655 0.40933158552360205 0.3932044049638235 0.351629230794864 1.0 0.3795403239310118 0.5056659866951355 0.4236558755007261 0.29224997080049936 0.38154464583114583 0.3588523335344297 0.41756751227673716 0.6899796025512063 0.6464231368641967 0.5433281730380624 0.562928918146957 0.477892920371326 0.5622906173252058 0.5251429952595528 0.5478722443132951 0.6049807731583691 0.6354652082204137 0.630402441122833 0.3794974255986497 0.3795403239310118 1.0 0.4930159416195486 0.5181201448630268 0.5803777383806147 0.6469214181468704 0.6948496765777495 0.4470859368404759 0.4829248204751438 0.5358162944590256 0.5184942908702888 0.548199461971616 0.5051290344693149 0.5224753114035746 0.5121609570322493 0.3931852360313254 0.4618381161019484 0.5592646909728172 0.46039334050493097 0.3853856901845631 0.5056659866951355 0.4930159416195486 1.0 0.42652705032056765 0.44932683446647737 0.43785942567407055 0.4753164362695061 0.4397187622064721 0.46889534799901644 0.48788180072620496 0.4221987742397929 0.4391780258991724 0.37501757322082124 0.40943675181474465 0.4009145053464104 0.46508600438409436 0.48763587982473167 0.4818543262464657 0.47890637662513413 0.3545226194665939 0.4236558755007261 0.5181201448630268 0.42652705032056765 1.0 0.47790886390032605 0.48725934120100245 0.5190064146136529 0.33202576165886044 0.8032247282972168 0.7422899354678594 0.7115418362708285 0.7360864919712798 0.7800445569351906 0.5093465044074202 0.5977517735470183 0.49382507887417354 0.7706072459010869 0.6472914883513033 0.585566235326388 0.2887247999104022 0.29224997080049936 0.5803777383806147 0.44932683446647737 0.47790886390032605 1.0 0.6969211415673917 0.7809706926931986 0.4888188579782026 0.683250978585417 0.5567944190945463 0.5950391891353299 0.5446831579286184 0.5544994546859855 0.5620661960809631 0.5250037933290096 0.567452359469588 0.7466598640213862 0.6601921339331994 0.6351507160237317 0.3457823701818523 0.38154464583114583 0.6469214181468704 0.43785942567407055 0.48725934120100245 0.6969211415673917 1.0 0.7728323266301629 0.48900850001764096 0.8070748362260224 0.665748475282032 0.6443046480828952 0.6929747243297233 0.6495006703990495 0.6186963081159329 0.5794376638957516 0.4917776473106598 0.9496984103271222 0.7498632642238646 0.689547274888211 0.3907912290635113 0.3588523335344297 0.6948496765777495 0.4753164362695061 0.5190064146136529 0.7809706926931986 0.7728323266301629 1.0 0.49523671133983144 0.5235913198639335 0.4888864809355612 0.4884301040402025 0.4930706471445408 0.5656995145637783 0.42757311673914256 0.3894891001523857 0.42276066378519034 0.5085284990098963 0.4817653659558023 0.4358153628530099 0.27865643358002923 0.41756751227673716 0.4470859368404759 0.4397187622064721 0.33202576165886044 0.4888188579782026 0.48900850001764096 0.49523671133983144 1.0],
    20,
    :,
)
corr ≈ testcorr