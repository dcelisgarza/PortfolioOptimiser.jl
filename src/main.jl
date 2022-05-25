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

mu = vec(ret_model(MRet(), Matrix(returns)))
S = risk_matrix(Cov(), Matrix(returns))

spy_prices = CSV.read("./test/assets/spy_prices.csv", DataFrame)
delta = market_implied_risk_aversion(spy_prices[!, 2])
deltatest = 2.685491066228311
delta ≈ deltatest

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
picking = hcat(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
)

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
testpost_cov = [
    0.05334874453292 0.024902672047444 0.035820877126815 0.03003762293178 0.040239757633216 0.015753746209349 0.020901320606043 0.010599290285722 0.022969634599781 0.018184452941796 0.010240781936638 0.025973258402243 0.009064297749123 0.013538208101076 0.006210715239239 0.012688234416409 0.025017676973642 0.01496297568946 0.020016194441763 0.021040600912291
    0.024902672047444 0.05378655346829 0.027715737668323 0.024028034151111 0.027240121725497 0.014780532959058 0.033098711831861 0.011552819915695 0.023921951319338 0.019261317981843 0.010624375436169 0.023173090065358 0.018121979504474 0.014968035434157 0.013747654871833 0.018860638211729 0.023508873867951 0.012939357937947 0.02174053531735 0.018137037509882
    0.035820877126815 0.027715737668323 0.065315687317338 0.030317625879565 0.038957855421918 0.017024676408514 0.029722501289929 0.00980478073907 0.023954242508466 0.019314377339082 0.009908592917906 0.03499366030552 0.014762971727342 0.013323558842692 0.010996319122858 0.014300277970999 0.025812084646399 0.013774651234565 0.020627276954368 0.022289714337395
    0.03003762293178 0.024028034151111 0.030317625879565 0.101918357017587 0.033684859429564 0.016546732063455 0.051026767291063 0.008792539122271 0.024821010516939 0.02070838698998 0.007265433614312 0.028160522674305 0.025507344746638 0.014305718590147 0.024501686107163 0.019770838251757 0.025634708805639 0.013261994393431 0.021428453897999 0.0175104551891
    0.040239757633216 0.027240121725497 0.038957855421918 0.033684859429564 0.084186441326594 0.013102039631964 0.030683114990461 0.009329532359921 0.022338707473804 0.016723425860667 0.009469164201705 0.030239737474177 0.004868424300702 0.012996415834839 0.019688863589902 0.014462589996749 0.026523493457822 0.011781491403383 0.019067922153415 0.025290726153949
    0.015753746209349 0.014780532959058 0.017024676408514 0.016546732063455 0.013102039631964 0.045546322618926 0.024053819130086 0.009721114482515 0.025348180325503 0.022066907386658 0.013407954548335 0.027118500000968 0.029271861523519 0.017666148809851 0.025616703559727 0.01211576330532 0.016440745985032 0.014279107809269 0.023222340135006 0.015556744457971
    0.020901320606043 0.033098711831861 0.029722501289929 0.051026767291063 0.030683114990461 0.024053819130086 0.419435740109041 0.011477565295854 0.042579570395332 0.035219728639183 0.016410203945159 0.026858724945831 0.035576381829818 0.02279674386249 0.061254442712256 0.030485215262191 0.033182344528116 0.018843469973971 0.031989975975234 0.013556548739957
    0.010599290285722 0.011552819915695 0.00980478073907 0.008792539122271 0.009329532359921 0.009721114482515 0.011477565295854 0.041603952007734 0.010508522499114 0.012654353044667 0.010775500238374 0.016264694466671 0.02839435328056 0.008767439301597 0.005296035825415 0.014670575655231 0.011681472878172 0.011308790475922 0.011995716652398 0.012350793595876
    0.022969634599781 0.023921951319338 0.023954242508466 0.024821010516939 0.022338707473804 0.025348180325503 0.042579570395332 0.010508522499114 0.070059650164303 0.031932681675489 0.011914877668708 0.035953252498342 0.037689078480604 0.023206827652217 0.03438071083178 0.030032807360198 0.027858949081679 0.01968588510039 0.051021491390749 0.019709534765143
    0.018184452941796 0.019261317981843 0.019314377339082 0.02070838698998 0.016723425860667 0.022066907386658 0.035219728639183 0.012654353044667 0.031932681675489 0.061876666057302 0.012119544562166 0.032548730414819 0.037031741557304 0.016678576740828 0.026459306692323 0.025405843850092 0.01951833438857 0.015293548042868 0.027455167376016 0.017700258458998
    0.010240781936638 0.010624375436169 0.009908592917906 0.007265433614312 0.009469164201705 0.013407954548335 0.016410203945159 0.010775500238374 0.011914877668708 0.012119544562166 0.026795081644734 0.012420626767153 0.014970292581857 0.012347504980603 0.015922535723888 0.010251092702776 0.010843846272749 0.009480106941065 0.01309306276484 0.00914149735524
    0.025973258402243 0.023173090065358 0.03499366030552 0.028160522674305 0.030239737474177 0.027118500000968 0.026858724945831 0.016264694466671 0.035953252498342 0.032548730414819 0.012420626767153 0.179749910435694 0.058493054783132 0.019223964528325 0.033173828688863 0.034920875598853 0.023589018847215 0.017337377334031 0.030907137690852 0.027376522902746
    0.009064297749123 0.018121979504474 0.014762971727342 0.025507344746638 0.004868424300702 0.029271861523519 0.035576381829818 0.02839435328056 0.037689078480604 0.037031741557304 0.014970292581857 0.058493054783132 0.494697541906723 0.019898343296291 0.056376935787249 0.059452838044951 0.015111514351748 0.014180557030523 0.025439788469831 0.023979222437031
    0.013538208101076 0.014968035434157 0.013323558842692 0.014305718590147 0.012996415834839 0.017666148809851 0.02279674386249 0.008767439301597 0.023206827652217 0.016678576740828 0.012347504980603 0.019223964528325 0.019898343296291 0.036539546031524 0.039313250146948 0.012002339660102 0.016367538036895 0.013024216890089 0.022285558956566 0.010168475613478
    0.006210715239239 0.013747654871833 0.010996319122858 0.024501686107163 0.019688863589902 0.025616703559727 0.061254442712256 0.005296035825415 0.03438071083178 0.026459306692323 0.015922535723888 0.033173828688863 0.056376935787249 0.039313250146948 0.252892111331342 0.029103381607604 0.01702654624531 0.00955948287177 0.0295607068289 0.008895620469753
    0.012688234416409 0.018860638211729 0.014300277970999 0.019770838251757 0.014462589996749 0.01211576330532 0.030485215262191 0.014670575655231 0.030032807360198 0.025405843850092 0.010251092702776 0.034920875598853 0.059452838044951 0.012002339660102 0.029103381607604 0.128579631489709 0.018970727664979 0.012512528618383 0.02346950763889 0.019336816446419
    0.025017676973642 0.023508873867951 0.025812084646399 0.025634708805639 0.026523493457822 0.016440745985032 0.033182344528116 0.011681472878172 0.027858949081679 0.01951833438857 0.010843846272749 0.023589018847215 0.015111514351748 0.016367538036895 0.01702654624531 0.018970727664979 0.041281485121884 0.015560942620389 0.024957639540371 0.019587608239884
    0.01496297568946 0.012939357937947 0.013774651234565 0.013261994393431 0.011781491403383 0.014279107809269 0.018843469973971 0.011308790475922 0.01968588510039 0.015293548042868 0.009480106941065 0.017337377334031 0.014180557030523 0.013024216890089 0.00955948287177 0.012512528618383 0.015560942620389 0.031037346425707 0.018369433119826 0.011288056098657
    0.020016194441763 0.02174053531735 0.020627276954368 0.021428453897999 0.019067922153415 0.023222340135006 0.031989975975234 0.011995716652398 0.051021491390749 0.027455167376016 0.01309306276484 0.030907137690852 0.025439788469831 0.022285558956566 0.0295607068289 0.02346950763889 0.024957639540371 0.018369433119826 0.046919986197744 0.01794461920481
    0.021040600912291 0.018137037509882 0.022289714337395 0.0175104551891 0.025290726153949 0.015556744457971 0.013556548739957 0.012350793595876 0.019709534765143 0.017700258458998 0.00914149735524 0.027376522902746 0.023979222437031 0.010168475613478 0.008895620469753 0.019336816446419 0.019587608239884 0.011288056098657 0.01794461920481 0.039986416856178
]
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

mu, sigma, sr = portfolio_performance(bl)
mutest, sigmatest, srtest = 1.0093606663118133, 1.405512109425161, 0.7039147223828978
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest

bl = BlackLitterman(mcapsdf[!, 1], S; rf = 0, tau = 0.01, Q = views, P = picking)
testpost_ret = [
    -0.020830526939649,
    -0.031015053178948,
    -0.072163030613922,
    -0.024603378927613,
    -0.047074290570558,
    -0.052593753584965,
    -0.011573284998936,
    -0.029853302581122,
    0.028940053197696,
    -0.023279044028087,
    -0.038689702120485,
    -0.054944226494396,
    -0.050783791594354,
    -0.011499423908914,
    -0.006534148187139,
    -0.022294481371801,
    -0.027596053560504,
    -0.014608054178002,
    0.012267692276885,
    -0.094287655854972,
]
bl.post_ret ≈ testpost_ret
testpost_cov = reshape(
    [
        0.05334874453292,
        0.024902672047444,
        0.035820877126815,
        0.03003762293178,
        0.040239757633216,
        0.015753746209349,
        0.020901320606043,
        0.010599290285722,
        0.022969634599781,
        0.018184452941796,
        0.010240781936638,
        0.025973258402243,
        0.009064297749123,
        0.013538208101076,
        0.006210715239239,
        0.012688234416409,
        0.025017676973642,
        0.01496297568946,
        0.020016194441763,
        0.021040600912291,
        0.024902672047444,
        0.05378655346829,
        0.027715737668323,
        0.024028034151111,
        0.027240121725497,
        0.014780532959058,
        0.033098711831861,
        0.011552819915695,
        0.023921951319338,
        0.019261317981843,
        0.010624375436169,
        0.023173090065358,
        0.018121979504474,
        0.014968035434157,
        0.013747654871833,
        0.018860638211729,
        0.023508873867951,
        0.012939357937947,
        0.02174053531735,
        0.018137037509882,
        0.035820877126815,
        0.027715737668323,
        0.065315687317338,
        0.030317625879565,
        0.038957855421918,
        0.017024676408514,
        0.029722501289929,
        0.00980478073907,
        0.023954242508466,
        0.019314377339082,
        0.009908592917906,
        0.03499366030552,
        0.014762971727342,
        0.013323558842692,
        0.010996319122858,
        0.014300277970999,
        0.025812084646399,
        0.013774651234565,
        0.020627276954368,
        0.022289714337395,
        0.03003762293178,
        0.024028034151111,
        0.030317625879565,
        0.101918357017587,
        0.033684859429564,
        0.016546732063455,
        0.051026767291063,
        0.008792539122271,
        0.024821010516939,
        0.02070838698998,
        0.007265433614312,
        0.028160522674305,
        0.025507344746638,
        0.014305718590147,
        0.024501686107163,
        0.019770838251757,
        0.025634708805639,
        0.013261994393431,
        0.021428453897999,
        0.0175104551891,
        0.040239757633216,
        0.027240121725497,
        0.038957855421918,
        0.033684859429564,
        0.084186441326594,
        0.013102039631964,
        0.030683114990461,
        0.009329532359921,
        0.022338707473804,
        0.016723425860667,
        0.009469164201705,
        0.030239737474177,
        0.004868424300702,
        0.012996415834839,
        0.019688863589902,
        0.014462589996749,
        0.026523493457822,
        0.011781491403383,
        0.019067922153415,
        0.025290726153949,
        0.015753746209349,
        0.014780532959058,
        0.017024676408514,
        0.016546732063455,
        0.013102039631964,
        0.045546322618926,
        0.024053819130086,
        0.009721114482515,
        0.025348180325503,
        0.022066907386658,
        0.013407954548335,
        0.027118500000968,
        0.029271861523519,
        0.017666148809851,
        0.025616703559727,
        0.01211576330532,
        0.016440745985032,
        0.014279107809269,
        0.023222340135006,
        0.015556744457971,
        0.020901320606043,
        0.033098711831861,
        0.029722501289929,
        0.051026767291063,
        0.030683114990461,
        0.024053819130086,
        0.419435740109041,
        0.011477565295854,
        0.042579570395332,
        0.035219728639183,
        0.016410203945159,
        0.026858724945831,
        0.035576381829818,
        0.02279674386249,
        0.061254442712256,
        0.030485215262191,
        0.033182344528116,
        0.018843469973971,
        0.031989975975234,
        0.013556548739957,
        0.010599290285722,
        0.011552819915695,
        0.00980478073907,
        0.008792539122271,
        0.009329532359921,
        0.009721114482515,
        0.011477565295854,
        0.041603952007734,
        0.010508522499114,
        0.012654353044667,
        0.010775500238374,
        0.016264694466671,
        0.02839435328056,
        0.008767439301597,
        0.005296035825415,
        0.014670575655231,
        0.011681472878172,
        0.011308790475922,
        0.011995716652398,
        0.012350793595876,
        0.022969634599781,
        0.023921951319338,
        0.023954242508466,
        0.024821010516939,
        0.022338707473804,
        0.025348180325503,
        0.042579570395332,
        0.010508522499114,
        0.070059650164303,
        0.031932681675489,
        0.011914877668708,
        0.035953252498342,
        0.037689078480604,
        0.023206827652217,
        0.03438071083178,
        0.030032807360198,
        0.027858949081679,
        0.01968588510039,
        0.051021491390749,
        0.019709534765143,
        0.018184452941796,
        0.019261317981843,
        0.019314377339082,
        0.02070838698998,
        0.016723425860667,
        0.022066907386658,
        0.035219728639183,
        0.012654353044667,
        0.031932681675489,
        0.061876666057302,
        0.012119544562166,
        0.032548730414819,
        0.037031741557304,
        0.016678576740828,
        0.026459306692323,
        0.025405843850092,
        0.01951833438857,
        0.015293548042868,
        0.027455167376016,
        0.017700258458998,
        0.010240781936638,
        0.010624375436169,
        0.009908592917906,
        0.007265433614312,
        0.009469164201705,
        0.013407954548335,
        0.016410203945159,
        0.010775500238374,
        0.011914877668708,
        0.012119544562166,
        0.026795081644734,
        0.012420626767153,
        0.014970292581857,
        0.012347504980603,
        0.015922535723888,
        0.010251092702776,
        0.010843846272749,
        0.009480106941065,
        0.01309306276484,
        0.00914149735524,
        0.025973258402243,
        0.023173090065358,
        0.03499366030552,
        0.028160522674305,
        0.030239737474177,
        0.027118500000968,
        0.026858724945831,
        0.016264694466671,
        0.035953252498342,
        0.032548730414819,
        0.012420626767153,
        0.179749910435694,
        0.058493054783132,
        0.019223964528325,
        0.033173828688863,
        0.034920875598853,
        0.023589018847215,
        0.017337377334031,
        0.030907137690852,
        0.027376522902746,
        0.009064297749123,
        0.018121979504474,
        0.014762971727342,
        0.025507344746638,
        0.004868424300702,
        0.029271861523519,
        0.035576381829818,
        0.02839435328056,
        0.037689078480604,
        0.037031741557304,
        0.014970292581857,
        0.058493054783132,
        0.494697541906723,
        0.019898343296291,
        0.056376935787249,
        0.059452838044951,
        0.015111514351748,
        0.014180557030523,
        0.025439788469831,
        0.023979222437031,
        0.013538208101076,
        0.014968035434157,
        0.013323558842692,
        0.014305718590147,
        0.012996415834839,
        0.017666148809851,
        0.02279674386249,
        0.008767439301597,
        0.023206827652217,
        0.016678576740828,
        0.012347504980603,
        0.019223964528325,
        0.019898343296291,
        0.036539546031524,
        0.039313250146948,
        0.012002339660102,
        0.016367538036895,
        0.013024216890089,
        0.022285558956566,
        0.010168475613478,
        0.006210715239239,
        0.013747654871833,
        0.010996319122858,
        0.024501686107163,
        0.019688863589902,
        0.025616703559727,
        0.061254442712256,
        0.005296035825415,
        0.03438071083178,
        0.026459306692323,
        0.015922535723888,
        0.033173828688863,
        0.056376935787249,
        0.039313250146948,
        0.252892111331342,
        0.029103381607604,
        0.01702654624531,
        0.00955948287177,
        0.0295607068289,
        0.008895620469753,
        0.012688234416409,
        0.018860638211729,
        0.014300277970999,
        0.019770838251757,
        0.014462589996749,
        0.01211576330532,
        0.030485215262191,
        0.014670575655231,
        0.030032807360198,
        0.025405843850092,
        0.010251092702776,
        0.034920875598853,
        0.059452838044951,
        0.012002339660102,
        0.029103381607604,
        0.128579631489709,
        0.018970727664979,
        0.012512528618383,
        0.02346950763889,
        0.019336816446419,
        0.025017676973642,
        0.023508873867951,
        0.025812084646399,
        0.025634708805639,
        0.026523493457822,
        0.016440745985032,
        0.033182344528116,
        0.011681472878172,
        0.027858949081679,
        0.01951833438857,
        0.010843846272749,
        0.023589018847215,
        0.015111514351748,
        0.016367538036895,
        0.01702654624531,
        0.018970727664979,
        0.041281485121884,
        0.015560942620389,
        0.024957639540371,
        0.019587608239884,
        0.01496297568946,
        0.012939357937947,
        0.013774651234565,
        0.013261994393431,
        0.011781491403383,
        0.014279107809269,
        0.018843469973971,
        0.011308790475922,
        0.01968588510039,
        0.015293548042868,
        0.009480106941065,
        0.017337377334031,
        0.014180557030523,
        0.013024216890089,
        0.00955948287177,
        0.012512528618383,
        0.015560942620389,
        0.031037346425707,
        0.018369433119826,
        0.011288056098657,
        0.020016194441763,
        0.02174053531735,
        0.020627276954368,
        0.021428453897999,
        0.019067922153415,
        0.023222340135006,
        0.031989975975234,
        0.011995716652398,
        0.051021491390749,
        0.027455167376016,
        0.01309306276484,
        0.030907137690852,
        0.025439788469831,
        0.022285558956566,
        0.0295607068289,
        0.02346950763889,
        0.024957639540371,
        0.018369433119826,
        0.046919986197744,
        0.01794461920481,
        0.021040600912291,
        0.018137037509882,
        0.022289714337395,
        0.0175104551891,
        0.025290726153949,
        0.015556744457971,
        0.013556548739957,
        0.012350793595876,
        0.019709534765143,
        0.017700258458998,
        0.00914149735524,
        0.027376522902746,
        0.023979222437031,
        0.010168475613478,
        0.008895620469753,
        0.019336816446419,
        0.019587608239884,
        0.011288056098657,
        0.01794461920481,
        0.039986416856178,
    ],
    20,
    :,
)
bl.post_cov ≈ testpost_cov
testblweights = [
    8.754664998496072,
    1.0732578926250225,
    -8.146850880828588,
    0.9455529695359096,
    0.35684387807620915,
    -6.916912407717437,
    -0.2775240638458601,
    4.035813523265117,
    8.142914952354905,
    0.5313680726647304,
    -0.3470878250829728,
    -0.6041848156310756,
    -0.2610045880233343,
    4.169128323738159,
    0.02210801019603272,
    0.7268002248416386,
    0.4163589038618726,
    5.818178434078446,
    6.619286979429875,
    -24.058712582034723,
]
bl.weights ≈ testblweights

mu, sigma, sr = portfolio_performance(bl)
mutest, sigmatest, srtest = -0.1634702972666502, 0.24870800299247436, -0.737693580661342
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest

bl = BlackLitterman(
    mcapsdf[!, 1],
    S;
    rf = 0,
    tau = 0.01,
    pi = :market, # either a vector, `nothing`, `:equal`, or `:market`
    market_caps = mcapsdf[!, 2],
    Q = views,
    P = picking,
)
testpost_ret = [
    0.003159709618849,
    -0.008198179020157,
    -0.047973127241007,
    0.001146509165171,
    -0.019298956843574,
    -0.038975395634745,
    0.015357849340789,
    -0.020698432886203,
    0.046307334847981,
    -0.009008922543199,
    -0.029397051577061,
    -0.034675794405371,
    -0.039366481896771,
    0.000978169150907,
    0.007641576415112,
    -0.011088705297456,
    -0.009702793037667,
    -0.003618163606929,
    0.027981726053541,
    -0.085034119105279,
]
bl.post_ret ≈ testpost_ret
testpost_cov = reshape(
    [
        0.05334874453292,
        0.024902672047444,
        0.035820877126815,
        0.03003762293178,
        0.040239757633216,
        0.015753746209349,
        0.020901320606043,
        0.010599290285722,
        0.022969634599781,
        0.018184452941796,
        0.010240781936638,
        0.025973258402243,
        0.009064297749123,
        0.013538208101076,
        0.006210715239239,
        0.012688234416409,
        0.025017676973642,
        0.01496297568946,
        0.020016194441763,
        0.021040600912291,
        0.024902672047444,
        0.05378655346829,
        0.027715737668323,
        0.024028034151111,
        0.027240121725497,
        0.014780532959058,
        0.033098711831861,
        0.011552819915695,
        0.023921951319338,
        0.019261317981843,
        0.010624375436169,
        0.023173090065358,
        0.018121979504474,
        0.014968035434157,
        0.013747654871833,
        0.018860638211729,
        0.023508873867951,
        0.012939357937947,
        0.02174053531735,
        0.018137037509882,
        0.035820877126815,
        0.027715737668323,
        0.065315687317338,
        0.030317625879565,
        0.038957855421918,
        0.017024676408514,
        0.029722501289929,
        0.00980478073907,
        0.023954242508466,
        0.019314377339082,
        0.009908592917906,
        0.03499366030552,
        0.014762971727342,
        0.013323558842692,
        0.010996319122858,
        0.014300277970999,
        0.025812084646399,
        0.013774651234565,
        0.020627276954368,
        0.022289714337395,
        0.03003762293178,
        0.024028034151111,
        0.030317625879565,
        0.101918357017587,
        0.033684859429564,
        0.016546732063455,
        0.051026767291063,
        0.008792539122271,
        0.024821010516939,
        0.02070838698998,
        0.007265433614312,
        0.028160522674305,
        0.025507344746638,
        0.014305718590147,
        0.024501686107163,
        0.019770838251757,
        0.025634708805639,
        0.013261994393431,
        0.021428453897999,
        0.0175104551891,
        0.040239757633216,
        0.027240121725497,
        0.038957855421918,
        0.033684859429564,
        0.084186441326594,
        0.013102039631964,
        0.030683114990461,
        0.009329532359921,
        0.022338707473804,
        0.016723425860667,
        0.009469164201705,
        0.030239737474177,
        0.004868424300702,
        0.012996415834839,
        0.019688863589902,
        0.014462589996749,
        0.026523493457822,
        0.011781491403383,
        0.019067922153415,
        0.025290726153949,
        0.015753746209349,
        0.014780532959058,
        0.017024676408514,
        0.016546732063455,
        0.013102039631964,
        0.045546322618926,
        0.024053819130086,
        0.009721114482515,
        0.025348180325503,
        0.022066907386658,
        0.013407954548335,
        0.027118500000968,
        0.029271861523519,
        0.017666148809851,
        0.025616703559727,
        0.01211576330532,
        0.016440745985032,
        0.014279107809269,
        0.023222340135006,
        0.015556744457971,
        0.020901320606043,
        0.033098711831861,
        0.029722501289929,
        0.051026767291063,
        0.030683114990461,
        0.024053819130086,
        0.419435740109041,
        0.011477565295854,
        0.042579570395332,
        0.035219728639183,
        0.016410203945159,
        0.026858724945831,
        0.035576381829818,
        0.02279674386249,
        0.061254442712256,
        0.030485215262191,
        0.033182344528116,
        0.018843469973971,
        0.031989975975234,
        0.013556548739957,
        0.010599290285722,
        0.011552819915695,
        0.00980478073907,
        0.008792539122271,
        0.009329532359921,
        0.009721114482515,
        0.011477565295854,
        0.041603952007734,
        0.010508522499114,
        0.012654353044667,
        0.010775500238374,
        0.016264694466671,
        0.02839435328056,
        0.008767439301597,
        0.005296035825415,
        0.014670575655231,
        0.011681472878172,
        0.011308790475922,
        0.011995716652398,
        0.012350793595876,
        0.022969634599781,
        0.023921951319338,
        0.023954242508466,
        0.024821010516939,
        0.022338707473804,
        0.025348180325503,
        0.042579570395332,
        0.010508522499114,
        0.070059650164303,
        0.031932681675489,
        0.011914877668708,
        0.035953252498342,
        0.037689078480604,
        0.023206827652217,
        0.03438071083178,
        0.030032807360198,
        0.027858949081679,
        0.01968588510039,
        0.051021491390749,
        0.019709534765143,
        0.018184452941796,
        0.019261317981843,
        0.019314377339082,
        0.02070838698998,
        0.016723425860667,
        0.022066907386658,
        0.035219728639183,
        0.012654353044667,
        0.031932681675489,
        0.061876666057302,
        0.012119544562166,
        0.032548730414819,
        0.037031741557304,
        0.016678576740828,
        0.026459306692323,
        0.025405843850092,
        0.01951833438857,
        0.015293548042868,
        0.027455167376016,
        0.017700258458998,
        0.010240781936638,
        0.010624375436169,
        0.009908592917906,
        0.007265433614312,
        0.009469164201705,
        0.013407954548335,
        0.016410203945159,
        0.010775500238374,
        0.011914877668708,
        0.012119544562166,
        0.026795081644734,
        0.012420626767153,
        0.014970292581857,
        0.012347504980603,
        0.015922535723888,
        0.010251092702776,
        0.010843846272749,
        0.009480106941065,
        0.01309306276484,
        0.00914149735524,
        0.025973258402243,
        0.023173090065358,
        0.03499366030552,
        0.028160522674305,
        0.030239737474177,
        0.027118500000968,
        0.026858724945831,
        0.016264694466671,
        0.035953252498342,
        0.032548730414819,
        0.012420626767153,
        0.179749910435694,
        0.058493054783132,
        0.019223964528325,
        0.033173828688863,
        0.034920875598853,
        0.023589018847215,
        0.017337377334031,
        0.030907137690852,
        0.027376522902746,
        0.009064297749123,
        0.018121979504474,
        0.014762971727342,
        0.025507344746638,
        0.004868424300702,
        0.029271861523519,
        0.035576381829818,
        0.02839435328056,
        0.037689078480604,
        0.037031741557304,
        0.014970292581857,
        0.058493054783132,
        0.494697541906723,
        0.019898343296291,
        0.056376935787249,
        0.059452838044951,
        0.015111514351748,
        0.014180557030523,
        0.025439788469831,
        0.023979222437031,
        0.013538208101076,
        0.014968035434157,
        0.013323558842692,
        0.014305718590147,
        0.012996415834839,
        0.017666148809851,
        0.02279674386249,
        0.008767439301597,
        0.023206827652217,
        0.016678576740828,
        0.012347504980603,
        0.019223964528325,
        0.019898343296291,
        0.036539546031524,
        0.039313250146948,
        0.012002339660102,
        0.016367538036895,
        0.013024216890089,
        0.022285558956566,
        0.010168475613478,
        0.006210715239239,
        0.013747654871833,
        0.010996319122858,
        0.024501686107163,
        0.019688863589902,
        0.025616703559727,
        0.061254442712256,
        0.005296035825415,
        0.03438071083178,
        0.026459306692323,
        0.015922535723888,
        0.033173828688863,
        0.056376935787249,
        0.039313250146948,
        0.252892111331342,
        0.029103381607604,
        0.01702654624531,
        0.00955948287177,
        0.0295607068289,
        0.008895620469753,
        0.012688234416409,
        0.018860638211729,
        0.014300277970999,
        0.019770838251757,
        0.014462589996749,
        0.01211576330532,
        0.030485215262191,
        0.014670575655231,
        0.030032807360198,
        0.025405843850092,
        0.010251092702776,
        0.034920875598853,
        0.059452838044951,
        0.012002339660102,
        0.029103381607604,
        0.128579631489709,
        0.018970727664979,
        0.012512528618383,
        0.02346950763889,
        0.019336816446419,
        0.025017676973642,
        0.023508873867951,
        0.025812084646399,
        0.025634708805639,
        0.026523493457822,
        0.016440745985032,
        0.033182344528116,
        0.011681472878172,
        0.027858949081679,
        0.01951833438857,
        0.010843846272749,
        0.023589018847215,
        0.015111514351748,
        0.016367538036895,
        0.01702654624531,
        0.018970727664979,
        0.041281485121884,
        0.015560942620389,
        0.024957639540371,
        0.019587608239884,
        0.01496297568946,
        0.012939357937947,
        0.013774651234565,
        0.013261994393431,
        0.011781491403383,
        0.014279107809269,
        0.018843469973971,
        0.011308790475922,
        0.01968588510039,
        0.015293548042868,
        0.009480106941065,
        0.017337377334031,
        0.014180557030523,
        0.013024216890089,
        0.00955948287177,
        0.012512528618383,
        0.015560942620389,
        0.031037346425707,
        0.018369433119826,
        0.011288056098657,
        0.020016194441763,
        0.02174053531735,
        0.020627276954368,
        0.021428453897999,
        0.019067922153415,
        0.023222340135006,
        0.031989975975234,
        0.011995716652398,
        0.051021491390749,
        0.027455167376016,
        0.01309306276484,
        0.030907137690852,
        0.025439788469831,
        0.022285558956566,
        0.0295607068289,
        0.02346950763889,
        0.024957639540371,
        0.018369433119826,
        0.046919986197744,
        0.01794461920481,
        0.021040600912291,
        0.018137037509882,
        0.022289714337395,
        0.0175104551891,
        0.025290726153949,
        0.015556744457971,
        0.013556548739957,
        0.012350793595876,
        0.019709534765143,
        0.017700258458998,
        0.00914149735524,
        0.027376522902746,
        0.023979222437031,
        0.010168475613478,
        0.008895620469753,
        0.019336816446419,
        0.019587608239884,
        0.011288056098657,
        0.01794461920481,
        0.039986416856178,
    ],
    20,
    :,
)
bl.post_cov ≈ testpost_cov

testblweights = [
    -0.6294139094485687,
    -0.09837977027829774,
    0.5053231235765305,
    -0.044064216435573664,
    -0.07167668977418844,
    0.48139995581445977,
    -0.003554899262156977,
    -0.028025833717935343,
    -0.5142207699325133,
    -0.004216275869069968,
    0.4842934784697037,
    -0.006448421917401054,
    -1.2305559159888653e-17,
    -0.024388262379914153,
    -8.267207586413058e-05,
    -0.001818785669010557,
    -0.023809557848865275,
    -0.01752648008319239,
    -0.5242240911120722,
    1.5208340779439298,
]
bl.weights ≈ testblweights

mu, sigma, sr = portfolio_performance(bl)
mutest, sigmatest, srtest = -0.22381742715934094, 0.3451253556188572, -0.7064604880222224
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest

bl = BlackLitterman(
    mcapsdf[!, 1],
    S;
    rf = 0,
    tau = 0.01,
    pi = :equal, # either a vector, `nothing`, `:equal`, or `:market`
    Q = views,
    P = picking,
)
testpost_ret = [
    0.016192023490508,
    0.008078079544674,
    -0.035526575723945,
    0.014915056431211,
    -0.012549200465761,
    -0.012617689405891,
    0.030779744648498,
    0.012417292831914,
    0.068538770170616,
    0.016192579991541,
    0.005195177139875,
    -0.021445679758055,
    -0.015375306432324,
    0.032468937295293,
    0.038440359581279,
    0.016292390683074,
    0.01069324078961,
    0.028619865249918,
    0.052446599909811,
    -0.069127469398986,
]
bl.post_ret ≈ testpost_ret
testpost_cov = reshape(
    [
        0.05334874453292,
        0.024902672047444,
        0.035820877126815,
        0.03003762293178,
        0.040239757633216,
        0.015753746209349,
        0.020901320606043,
        0.010599290285722,
        0.022969634599781,
        0.018184452941796,
        0.010240781936638,
        0.025973258402243,
        0.009064297749123,
        0.013538208101076,
        0.006210715239239,
        0.012688234416409,
        0.025017676973642,
        0.01496297568946,
        0.020016194441763,
        0.021040600912291,
        0.024902672047444,
        0.05378655346829,
        0.027715737668323,
        0.024028034151111,
        0.027240121725497,
        0.014780532959058,
        0.033098711831861,
        0.011552819915695,
        0.023921951319338,
        0.019261317981843,
        0.010624375436169,
        0.023173090065358,
        0.018121979504474,
        0.014968035434157,
        0.013747654871833,
        0.018860638211729,
        0.023508873867951,
        0.012939357937947,
        0.02174053531735,
        0.018137037509882,
        0.035820877126815,
        0.027715737668323,
        0.065315687317338,
        0.030317625879565,
        0.038957855421918,
        0.017024676408514,
        0.029722501289929,
        0.00980478073907,
        0.023954242508466,
        0.019314377339082,
        0.009908592917906,
        0.03499366030552,
        0.014762971727342,
        0.013323558842692,
        0.010996319122858,
        0.014300277970999,
        0.025812084646399,
        0.013774651234565,
        0.020627276954368,
        0.022289714337395,
        0.03003762293178,
        0.024028034151111,
        0.030317625879565,
        0.101918357017587,
        0.033684859429564,
        0.016546732063455,
        0.051026767291063,
        0.008792539122271,
        0.024821010516939,
        0.02070838698998,
        0.007265433614312,
        0.028160522674305,
        0.025507344746638,
        0.014305718590147,
        0.024501686107163,
        0.019770838251757,
        0.025634708805639,
        0.013261994393431,
        0.021428453897999,
        0.0175104551891,
        0.040239757633216,
        0.027240121725497,
        0.038957855421918,
        0.033684859429564,
        0.084186441326594,
        0.013102039631964,
        0.030683114990461,
        0.009329532359921,
        0.022338707473804,
        0.016723425860667,
        0.009469164201705,
        0.030239737474177,
        0.004868424300702,
        0.012996415834839,
        0.019688863589902,
        0.014462589996749,
        0.026523493457822,
        0.011781491403383,
        0.019067922153415,
        0.025290726153949,
        0.015753746209349,
        0.014780532959058,
        0.017024676408514,
        0.016546732063455,
        0.013102039631964,
        0.045546322618926,
        0.024053819130086,
        0.009721114482515,
        0.025348180325503,
        0.022066907386658,
        0.013407954548335,
        0.027118500000968,
        0.029271861523519,
        0.017666148809851,
        0.025616703559727,
        0.01211576330532,
        0.016440745985032,
        0.014279107809269,
        0.023222340135006,
        0.015556744457971,
        0.020901320606043,
        0.033098711831861,
        0.029722501289929,
        0.051026767291063,
        0.030683114990461,
        0.024053819130086,
        0.419435740109041,
        0.011477565295854,
        0.042579570395332,
        0.035219728639183,
        0.016410203945159,
        0.026858724945831,
        0.035576381829818,
        0.02279674386249,
        0.061254442712256,
        0.030485215262191,
        0.033182344528116,
        0.018843469973971,
        0.031989975975234,
        0.013556548739957,
        0.010599290285722,
        0.011552819915695,
        0.00980478073907,
        0.008792539122271,
        0.009329532359921,
        0.009721114482515,
        0.011477565295854,
        0.041603952007734,
        0.010508522499114,
        0.012654353044667,
        0.010775500238374,
        0.016264694466671,
        0.02839435328056,
        0.008767439301597,
        0.005296035825415,
        0.014670575655231,
        0.011681472878172,
        0.011308790475922,
        0.011995716652398,
        0.012350793595876,
        0.022969634599781,
        0.023921951319338,
        0.023954242508466,
        0.024821010516939,
        0.022338707473804,
        0.025348180325503,
        0.042579570395332,
        0.010508522499114,
        0.070059650164303,
        0.031932681675489,
        0.011914877668708,
        0.035953252498342,
        0.037689078480604,
        0.023206827652217,
        0.03438071083178,
        0.030032807360198,
        0.027858949081679,
        0.01968588510039,
        0.051021491390749,
        0.019709534765143,
        0.018184452941796,
        0.019261317981843,
        0.019314377339082,
        0.02070838698998,
        0.016723425860667,
        0.022066907386658,
        0.035219728639183,
        0.012654353044667,
        0.031932681675489,
        0.061876666057302,
        0.012119544562166,
        0.032548730414819,
        0.037031741557304,
        0.016678576740828,
        0.026459306692323,
        0.025405843850092,
        0.01951833438857,
        0.015293548042868,
        0.027455167376016,
        0.017700258458998,
        0.010240781936638,
        0.010624375436169,
        0.009908592917906,
        0.007265433614312,
        0.009469164201705,
        0.013407954548335,
        0.016410203945159,
        0.010775500238374,
        0.011914877668708,
        0.012119544562166,
        0.026795081644734,
        0.012420626767153,
        0.014970292581857,
        0.012347504980603,
        0.015922535723888,
        0.010251092702776,
        0.010843846272749,
        0.009480106941065,
        0.01309306276484,
        0.00914149735524,
        0.025973258402243,
        0.023173090065358,
        0.03499366030552,
        0.028160522674305,
        0.030239737474177,
        0.027118500000968,
        0.026858724945831,
        0.016264694466671,
        0.035953252498342,
        0.032548730414819,
        0.012420626767153,
        0.179749910435694,
        0.058493054783132,
        0.019223964528325,
        0.033173828688863,
        0.034920875598853,
        0.023589018847215,
        0.017337377334031,
        0.030907137690852,
        0.027376522902746,
        0.009064297749123,
        0.018121979504474,
        0.014762971727342,
        0.025507344746638,
        0.004868424300702,
        0.029271861523519,
        0.035576381829818,
        0.02839435328056,
        0.037689078480604,
        0.037031741557304,
        0.014970292581857,
        0.058493054783132,
        0.494697541906723,
        0.019898343296291,
        0.056376935787249,
        0.059452838044951,
        0.015111514351748,
        0.014180557030523,
        0.025439788469831,
        0.023979222437031,
        0.013538208101076,
        0.014968035434157,
        0.013323558842692,
        0.014305718590147,
        0.012996415834839,
        0.017666148809851,
        0.02279674386249,
        0.008767439301597,
        0.023206827652217,
        0.016678576740828,
        0.012347504980603,
        0.019223964528325,
        0.019898343296291,
        0.036539546031524,
        0.039313250146948,
        0.012002339660102,
        0.016367538036895,
        0.013024216890089,
        0.022285558956566,
        0.010168475613478,
        0.006210715239239,
        0.013747654871833,
        0.010996319122858,
        0.024501686107163,
        0.019688863589902,
        0.025616703559727,
        0.061254442712256,
        0.005296035825415,
        0.03438071083178,
        0.026459306692323,
        0.015922535723888,
        0.033173828688863,
        0.056376935787249,
        0.039313250146948,
        0.252892111331342,
        0.029103381607604,
        0.01702654624531,
        0.00955948287177,
        0.0295607068289,
        0.008895620469753,
        0.012688234416409,
        0.018860638211729,
        0.014300277970999,
        0.019770838251757,
        0.014462589996749,
        0.01211576330532,
        0.030485215262191,
        0.014670575655231,
        0.030032807360198,
        0.025405843850092,
        0.010251092702776,
        0.034920875598853,
        0.059452838044951,
        0.012002339660102,
        0.029103381607604,
        0.128579631489709,
        0.018970727664979,
        0.012512528618383,
        0.02346950763889,
        0.019336816446419,
        0.025017676973642,
        0.023508873867951,
        0.025812084646399,
        0.025634708805639,
        0.026523493457822,
        0.016440745985032,
        0.033182344528116,
        0.011681472878172,
        0.027858949081679,
        0.01951833438857,
        0.010843846272749,
        0.023589018847215,
        0.015111514351748,
        0.016367538036895,
        0.01702654624531,
        0.018970727664979,
        0.041281485121884,
        0.015560942620389,
        0.024957639540371,
        0.019587608239884,
        0.01496297568946,
        0.012939357937947,
        0.013774651234565,
        0.013261994393431,
        0.011781491403383,
        0.014279107809269,
        0.018843469973971,
        0.011308790475922,
        0.01968588510039,
        0.015293548042868,
        0.009480106941065,
        0.017337377334031,
        0.014180557030523,
        0.013024216890089,
        0.00955948287177,
        0.012512528618383,
        0.015560942620389,
        0.031037346425707,
        0.018369433119826,
        0.011288056098657,
        0.020016194441763,
        0.02174053531735,
        0.020627276954368,
        0.021428453897999,
        0.019067922153415,
        0.023222340135006,
        0.031989975975234,
        0.011995716652398,
        0.051021491390749,
        0.027455167376016,
        0.01309306276484,
        0.030907137690852,
        0.025439788469831,
        0.022285558956566,
        0.0295607068289,
        0.02346950763889,
        0.024957639540371,
        0.018369433119826,
        0.046919986197744,
        0.01794461920481,
        0.021040600912291,
        0.018137037509882,
        0.022289714337395,
        0.0175104551891,
        0.025290726153949,
        0.015556744457971,
        0.013556548739957,
        0.012350793595876,
        0.019709534765143,
        0.017700258458998,
        0.00914149735524,
        0.027376522902746,
        0.023979222437031,
        0.010168475613478,
        0.008895620469753,
        0.019336816446419,
        0.019587608239884,
        0.011288056098657,
        0.01794461920481,
        0.039986416856178,
    ],
    20,
    :,
)
bl.post_cov ≈ testpost_cov
testblweights = [
    8.754664998496072,
    1.0732578926250225,
    -8.146850880828588,
    0.9455529695359096,
    0.35684387807620915,
    -6.916912407717437,
    -0.2775240638458601,
    4.035813523265117,
    8.142914952354905,
    0.5313680726647304,
    -0.3470878250829728,
    -0.6041848156310756,
    -0.2610045880233343,
    4.169128323738159,
    0.02210801019603272,
    0.7268002248416386,
    0.4163589038618726,
    5.818178434078446,
    6.619286979429875,
    -24.058712582034723,
]
bl.weights ≈ testblweights

mu, sigma, sr = portfolio_performance(bl)
mutest, sigmatest, srtest = 3.489507035125494, 5.426654465092965, 0.639345485776393
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest

bl = BlackLitterman(
    mcapsdf[!, 1],
    S;
    rf = 0,
    tau = 0.01,
    pi = prior, # either a vector, `nothing`, `:equal`, or `:market`
    Q = views,
    P = picking,
    omega = :idzorek,
    view_confidence = ones(length(views)),
)
testpost_ret = [
    0.013244409305835,
    -0.003862569869929,
    -0.086755590694165,
    0.017883535855483,
    -0.033667602107403,
    -0.072841404411129,
    0.055401879433333,
    -0.032909672787738,
    0.108841692163082,
    -0.010074729322361,
    -0.04816087126435,
    -0.070450930614597,
    -0.082521192117745,
    0.018048332665877,
    0.035883001205855,
    -0.017763252145557,
    -0.011710954813223,
    0.005937523530032,
    0.070156032161439,
    -0.2,
]
bl.post_ret ≈ testpost_ret
testpost_cov = reshape(
    [
        0.053256543534734,
        0.02485717955359,
        0.035817619714355,
        0.029988307958865,
        0.040168108898735,
        0.015718391524181,
        0.020876469348865,
        0.010565499901285,
        0.022902978091904,
        0.018135729311071,
        0.010219823816128,
        0.025914655077702,
        0.00900971729413,
        0.013508308774706,
        0.00619243513112,
        0.012634980171098,
        0.024963364434361,
        0.014928533869611,
        0.019958071938655,
        0.020937329618582,
        0.02485717955359,
        0.05374167993442,
        0.027657768273465,
        0.023984202855246,
        0.027181921371324,
        0.014752012532679,
        0.033057137474918,
        0.011526953764051,
        0.023851942232172,
        0.019215953190019,
        0.010611027043246,
        0.023105515165171,
        0.018065012681861,
        0.014941801097853,
        0.013720572956834,
        0.018809737491406,
        0.023459592167097,
        0.012911443094151,
        0.021682003893448,
        0.018050243391523,
        0.035817619714355,
        0.027657768273465,
        0.065167479878108,
        0.030268405584729,
        0.038894644577928,
        0.016984117668096,
        0.029655017559715,
        0.009775901621619,
        0.023884750870884,
        0.019261551919389,
        0.0098898856717,
        0.034892444631564,
        0.01468199260245,
        0.013295608051251,
        0.010955054109048,
        0.01424068742045,
        0.025755486408541,
        0.013747080867653,
        0.020567895494028,
        0.022183370592067,
        0.029988307958865,
        0.023984202855246,
        0.030268405584729,
        0.101874666444492,
        0.033627442692993,
        0.016520905849791,
        0.050986320124768,
        0.008767555815056,
        0.024746899622433,
        0.020663198448029,
        0.007254070800019,
        0.028096036381827,
        0.025452880281545,
        0.014279280118194,
        0.024475075669774,
        0.019719957310083,
        0.025585679565071,
        0.01323381926109,
        0.02136722512479,
        0.017427045522604,
        0.040168108898735,
        0.027181921371324,
        0.038894644577928,
        0.033627442692993,
        0.084105839020058,
        0.013057482625446,
        0.030637508261343,
        0.009291592752752,
        0.022262308241168,
        0.016665094831991,
        0.009444553325608,
        0.030153806152355,
        0.004793607459318,
        0.012962474714869,
        0.019658669189534,
        0.014398229617691,
        0.026459303726848,
        0.01174415277131,
        0.019001285453577,
        0.025167580696414,
        0.015718391524181,
        0.014752012532679,
        0.016984117668096,
        0.016520905849791,
        0.013057482625446,
        0.045506878679339,
        0.024042375783768,
        0.009695466589876,
        0.025350821099752,
        0.022042116626904,
        0.013379911681984,
        0.027073395681668,
        0.029229753229419,
        0.017652502053035,
        0.025609267652104,
        0.012090611453685,
        0.016412061810848,
        0.01426245093153,
        0.023216799227331,
        0.015476976487021,
        0.020876469348865,
        0.033057137474918,
        0.029655017559715,
        0.050986320124768,
        0.030637508261343,
        0.024042375783768,
        0.4193830002235,
        0.011461456268777,
        0.042484455503263,
        0.035176002954457,
        0.016412020302663,
        0.026794578956909,
        0.035526316803062,
        0.02277153148325,
        0.06122068140341,
        0.030434104912454,
        0.033136321916934,
        0.018818716990605,
        0.031915915280527,
        0.013495696064937,
        0.010565499901285,
        0.011526953764051,
        0.009775901621619,
        0.008767555815056,
        0.009291592752752,
        0.009695466589876,
        0.011461456268777,
        0.04158454584264,
        0.010487452615155,
        0.012629520896697,
        0.010759192929477,
        0.016225956041488,
        0.028359504276614,
        0.00875316748315,
        0.005285322802455,
        0.014643930169031,
        0.011653647061782,
        0.011292466926478,
        0.011974748241831,
        0.012289284929863,
        0.022902978091904,
        0.023851942232172,
        0.023884750870884,
        0.024746899622433,
        0.022262308241168,
        0.025350821099752,
        0.042484455503263,
        0.010487452615155,
        0.06984396533883,
        0.031852079511833,
        0.011936960498535,
        0.0358549656547,
        0.037614366268123,
        0.023158256442708,
        0.03431823986322,
        0.029936615391661,
        0.027775071949442,
        0.019638479938432,
        0.05085828933697,
        0.019626763959626,
        0.018135729311071,
        0.019215953190019,
        0.019261551919389,
        0.020663198448029,
        0.016665094831991,
        0.022042116626904,
        0.035176002954457,
        0.012629520896697,
        0.031852079511833,
        0.061829683957123,
        0.012109622752937,
        0.032481770120198,
        0.036975714071886,
        0.016651103344083,
        0.026430631981669,
        0.025352677987965,
        0.019467551736462,
        0.01526454219229,
        0.02738924405632,
        0.017616476279587,
        0.010219823816128,
        0.010611027043246,
        0.0098898856717,
        0.007254070800019,
        0.009444553325608,
        0.013379911681984,
        0.016412020302663,
        0.010759192929477,
        0.011936960498535,
        0.012109622752937,
        0.026773113016096,
        0.012398910279535,
        0.014948102089414,
        0.012342333691718,
        0.015923663751803,
        0.010242425513935,
        0.010831430166694,
        0.009472576538247,
        0.013104413743994,
        0.009092834755815,
        0.025914655077702,
        0.023105515165171,
        0.034892444631564,
        0.028096036381827,
        0.030153806152355,
        0.027073395681668,
        0.026794578956909,
        0.016225956041488,
        0.0358549656547,
        0.032481770120198,
        0.012398910279535,
        0.179645407437457,
        0.058405486814376,
        0.019185773101533,
        0.033132546846781,
        0.034845867497757,
        0.023516303627135,
        0.017296997441985,
        0.03082419361403,
        0.027245208470164,
        0.00900971729413,
        0.018065012681861,
        0.01468199260245,
        0.025452880281545,
        0.004793607459318,
        0.029229753229419,
        0.035526316803062,
        0.028359504276614,
        0.037614366268123,
        0.036975714071886,
        0.014948102089414,
        0.058405486814376,
        0.494623032427318,
        0.019866350492007,
        0.056344563544883,
        0.059390642511711,
        0.015050259713064,
        0.014146146019123,
        0.025375185644141,
        0.023863106816131,
        0.013508308774706,
        0.014941801097853,
        0.013295608051251,
        0.014279280118194,
        0.012962474714869,
        0.017652502053035,
        0.02277153148325,
        0.00875316748315,
        0.023158256442708,
        0.016651103344083,
        0.012342333691718,
        0.019185773101533,
        0.019866350492007,
        0.036523375772781,
        0.039296623530467,
        0.011971193645525,
        0.016337859983078,
        0.013007124718704,
        0.022246046069868,
        0.010120477342456,
        0.00619243513112,
        0.013720572956834,
        0.010955054109048,
        0.024475075669774,
        0.019658669189534,
        0.025609267652104,
        0.06122068140341,
        0.005285322802455,
        0.03431823986322,
        0.026430631981669,
        0.015923663751803,
        0.033132546846781,
        0.056344563544883,
        0.039296623530467,
        0.252870406117809,
        0.029069899802156,
        0.016996319777651,
        0.009543050167162,
        0.029512038916943,
        0.008855628861118,
        0.012634980171098,
        0.018809737491406,
        0.01424068742045,
        0.019719957310083,
        0.014398229617691,
        0.012090611453685,
        0.030434104912454,
        0.014643930169031,
        0.029936615391661,
        0.025352677987965,
        0.010242425513935,
        0.034845867497757,
        0.059390642511711,
        0.011971193645525,
        0.029069899802156,
        0.12851913088368,
        0.018913489429945,
        0.012479936990966,
        0.023391698251987,
        0.019245993632443,
        0.024963364434361,
        0.023459592167097,
        0.025755486408541,
        0.025585679565071,
        0.026459303726848,
        0.016412061810848,
        0.033136321916934,
        0.011653647061782,
        0.027775071949442,
        0.019467551736462,
        0.010831430166694,
        0.023516303627135,
        0.015050259713064,
        0.016337859983078,
        0.016996319777651,
        0.018913489429945,
        0.04122643526786,
        0.015529396610032,
        0.024888466413595,
        0.019494417377056,
        0.014928533869611,
        0.012911443094151,
        0.013747080867653,
        0.01323381926109,
        0.01174415277131,
        0.01426245093153,
        0.018818716990605,
        0.011292466926478,
        0.019638479938432,
        0.01526454219229,
        0.009472576538247,
        0.017296997441985,
        0.014146146019123,
        0.013007124718704,
        0.009543050167162,
        0.012479936990966,
        0.015529396610032,
        0.031018977571267,
        0.018330186492931,
        0.01123416557741,
        0.019958071938655,
        0.021682003893448,
        0.020567895494028,
        0.02136722512479,
        0.019001285453577,
        0.023216799227331,
        0.031915915280527,
        0.011974748241831,
        0.05085828933697,
        0.02738924405632,
        0.013104413743994,
        0.03082419361403,
        0.025375185644141,
        0.022246046069868,
        0.029512038916943,
        0.023391698251987,
        0.024888466413595,
        0.018330186492931,
        0.046795038778593,
        0.017866465411268,
        0.020937329618582,
        0.018050243391523,
        0.022183370592067,
        0.017427045522604,
        0.025167580696414,
        0.015476976487021,
        0.013495696064937,
        0.012289284929863,
        0.019626763959626,
        0.017616476279587,
        0.009092834755815,
        0.027245208470164,
        0.023863106816131,
        0.010120477342456,
        0.008855628861118,
        0.019245993632443,
        0.019494417377056,
        0.01123416557741,
        0.017866465411268,
        0.039788747840675,
    ],
    20,
    :,
)
bl.post_cov ≈ testpost_cov
testblweights = [
    -0.7158337577632388,
    -0.16394497820881257,
    0.5196428529940469,
    -0.0800365157310615,
    -0.11340466722869529,
    0.5513553465965026,
    -0.0014291563205649168,
    -0.0997103418236338,
    -0.6183837108784705,
    -0.013923650510731987,
    0.4623672632358077,
    -0.0011604300339745556,
    0.0037090423814285746,
    -0.09610692714812996,
    -0.0004391213357471932,
    -0.013077248436743842,
    -0.04190300342022579,
    -0.10916994048869978,
    -0.6118512157359796,
    2.143300159856924,
]
bl.weights ≈ testblweights
mu, sigma, sr = portfolio_performance(bl)
mutest, sigmatest, srtest = -0.6514205594222443, 0.4406932807032015, -1.5235552453871726
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest

bl = BlackLitterman(
    mcapsdf[!, 1],
    S;
    rf = 0,
    tau = 0.01,
    pi = prior, # either a vector, `nothing`, `:equal`, or `:market`
    Q = 1:20,
)
testpost_ret = [
    5.545949766202965,
    5.859922942637,
    5.695664060619475,
    5.703366392447494,
    6.126054128156325,
    8.163386700504887,
    8.941784495417364,
    7.552737233564954,
    12.865061946493816,
    9.944609532944712,
    8.095426708935904,
    11.819782830480356,
    12.788234028289967,
    10.342157130119974,
    12.758522515641998,
    12.557536835380693,
    11.57620566362041,
    11.341454567600675,
    12.960756826899697,
    11.825159038593647,
]
bl.post_ret ≈ testpost_ret
testpost_cov = reshape(
    [
        0.053107715144959,
        0.024730828691125,
        0.035531529948162,
        0.029830373487619,
        0.039986326618218,
        0.015639934661763,
        0.020717051997126,
        0.010533955627072,
        0.022835850857602,
        0.018066933301929,
        0.010164011454023,
        0.025792623985477,
        0.009014767336743,
        0.013444094089858,
        0.006145314972987,
        0.012617422059613,
        0.024855770454683,
        0.014865667824256,
        0.019895455602401,
        0.020958081509075,
        0.024730828691125,
        0.053521197789454,
        0.027539252047971,
        0.023859086547918,
        0.027061654508764,
        0.01466781161729,
        0.032855137771222,
        0.011475380468296,
        0.023783404389112,
        0.019134448479461,
        0.010538772157129,
        0.023025640295055,
        0.018010356273447,
        0.014861386563498,
        0.013638319411314,
        0.018747646272833,
        0.023357024220627,
        0.012848910366333,
        0.021611175750742,
        0.018065153819997,
        0.035531529948162,
        0.027539252047971,
        0.065075293468246,
        0.030104780167528,
        0.038698297029654,
        0.016907587517012,
        0.029524649585665,
        0.00973968027519,
        0.023816224603786,
        0.019192528405359,
        0.009831174592226,
        0.034800165577923,
        0.014701843300455,
        0.013227346058564,
        0.010923679486771,
        0.014224744939684,
        0.025647022447337,
        0.01367692747826,
        0.020503772897029,
        0.022205670397415,
        0.029830373487619,
        0.023859086547918,
        0.030104780167528,
        0.1014040039534,
        0.033453861193512,
        0.016418272123084,
        0.050648543439113,
        0.008732327976382,
        0.024673976170805,
        0.020567350447689,
        0.007198622893348,
        0.027970812078804,
        0.025341018683828,
        0.014199498612699,
        0.024315657371563,
        0.019647353283982,
        0.025462744213296,
        0.013168340447608,
        0.021298192972989,
        0.017434638452365,
        0.039986326618218,
        0.027061654508764,
        0.038698297029654,
        0.033453861193512,
        0.083773718612483,
        0.013011285016265,
        0.030455739415248,
        0.009276620716371,
        0.022219981525571,
        0.016623216391689,
        0.009399946342194,
        0.03005741917476,
        0.004867098157072,
        0.012908150108025,
        0.019556059462447,
        0.014391526088137,
        0.026357729073912,
        0.011705463134747,
        0.018962571011148,
        0.025201263005006,
        0.015639934661763,
        0.01466781161729,
        0.016907587517012,
        0.016418272123084,
        0.013011285016265,
        0.045315428358277,
        0.023843460704715,
        0.00965273391392,
        0.025126115301022,
        0.02189888474018,
        0.013316963877657,
        0.026922248846821,
        0.029057096349843,
        0.017525841598194,
        0.025400081624227,
        0.012015868201686,
        0.016315513326883,
        0.014168835136499,
        0.023028308496324,
        0.015492581331411,
        0.020717051997126,
        0.032855137771222,
        0.029524649585665,
        0.050648543439113,
        0.030455739415248,
        0.023843460704715,
        0.417314336046769,
        0.011383575320319,
        0.042307740500053,
        0.034959071158377,
        0.016260652948851,
        0.026658922097204,
        0.035301825771737,
        0.022610380300923,
        0.060782559691177,
        0.030267176723943,
        0.032942391574749,
        0.018696913763589,
        0.031770755867088,
        0.013480095835065,
        0.010533955627072,
        0.011475380468296,
        0.00973968027519,
        0.008732327976382,
        0.009276620716371,
        0.00965273391392,
        0.011383575320319,
        0.041399674920069,
        0.010429329987809,
        0.012566313632141,
        0.0107000555266,
        0.016156988740726,
        0.028197212655024,
        0.008701748581389,
        0.005244642491751,
        0.014571354251844,
        0.011604999561805,
        0.01122982868689,
        0.011911672152197,
        0.012304545525059,
        0.022835850857602,
        0.023783404389112,
        0.023816224603786,
        0.024673976170805,
        0.022219981525571,
        0.025126115301022,
        0.042307740500053,
        0.010429329987809,
        0.069826950418898,
        0.031745141472827,
        0.011778819476054,
        0.035743727915929,
        0.037449512967382,
        0.023053095042842,
        0.034140354072037,
        0.029881062967682,
        0.027708201693871,
        0.019563439481637,
        0.050818823024488,
        0.019635934227587,
        0.018066933301929,
        0.019134448479461,
        0.019192528405359,
        0.020567350447689,
        0.016623216391689,
        0.02189888474018,
        0.034959071158377,
        0.012566313632141,
        0.031745141472827,
        0.061570437921302,
        0.012017825688885,
        0.032334547289324,
        0.036771140638982,
        0.016554858983314,
        0.026253066856276,
        0.025243458444631,
        0.019392642391036,
        0.01518667984023,
        0.027288645057825,
        0.017631040262766,
        0.010164011454023,
        0.010538772157129,
        0.009831174592226,
        0.007198622893348,
        0.009399946342194,
        0.013316963877657,
        0.016260652948851,
        0.0107000555266,
        0.011778819476054,
        0.012017825688885,
        0.026664881536559,
        0.012318890939658,
        0.014850289677562,
        0.012246846533265,
        0.015781114742522,
        0.010164232376223,
        0.010754787546861,
        0.009402631911497,
        0.012965678635229,
        0.009099063413323,
        0.025792623985477,
        0.023025640295055,
        0.034800165577923,
        0.027970812078804,
        0.03005741917476,
        0.026922248846821,
        0.026658922097204,
        0.016156988740726,
        0.035743727915929,
        0.032334547289324,
        0.012318890939658,
        0.178890814297857,
        0.058095164831709,
        0.019085542653167,
        0.032924691139784,
        0.034701659654572,
        0.023442556259673,
        0.017219320577697,
        0.030722972905583,
        0.027274607535281,
        0.009014767336743,
        0.018010356273447,
        0.014701843300455,
        0.025341018683828,
        0.004867098157072,
        0.029057096349843,
        0.035301825771737,
        0.028197212655024,
        0.037449512967382,
        0.036771140638982,
        0.014850289677562,
        0.058095164831709,
        0.49221718403788,
        0.019747293677929,
        0.055942633881235,
        0.059038359088583,
        0.01501947949802,
        0.014078650283102,
        0.025265729520182,
        0.023889942860297,
        0.013444094089858,
        0.014861386563498,
        0.013227346058564,
        0.014199498612699,
        0.012908150108025,
        0.017525841598194,
        0.022610380300923,
        0.008701748581389,
        0.023053095042842,
        0.016554858983314,
        0.012246846533265,
        0.019085542653167,
        0.019747293677929,
        0.036341137066081,
        0.039023150015569,
        0.011917892973367,
        0.016253657497603,
        0.012928543934715,
        0.022137964432179,
        0.010120821228927,
        0.006145314972987,
        0.013638319411314,
        0.010923679486771,
        0.024315657371563,
        0.019556059462447,
        0.025400081624227,
        0.060782559691177,
        0.005244642491751,
        0.034140354072037,
        0.026253066856276,
        0.015781114742522,
        0.032924691139784,
        0.055942633881235,
        0.039023150015569,
        0.251570998399791,
        0.028889623593594,
        0.016895705412684,
        0.00947457479053,
        0.029347817041573,
        0.00884477964579,
        0.012617422059613,
        0.018747646272833,
        0.014224744939684,
        0.019647353283982,
        0.014391526088137,
        0.012015868201686,
        0.030267176723943,
        0.014571354251844,
        0.029881062967682,
        0.025243458444631,
        0.010164232376223,
        0.034701659654572,
        0.059038359088583,
        0.011917892973367,
        0.028889623593594,
        0.127964796432385,
        0.01886260826048,
        0.012432166798258,
        0.0233452842333,
        0.01926890291218,
        0.024855770454683,
        0.023357024220627,
        0.025647022447337,
        0.025462744213296,
        0.026357729073912,
        0.016315513326883,
        0.032942391574749,
        0.011604999561805,
        0.027708201693871,
        0.019392642391036,
        0.010754787546861,
        0.023442556259673,
        0.01501947949802,
        0.016253657497603,
        0.016895705412684,
        0.01886260826048,
        0.041081721004908,
        0.015457370217179,
        0.02481757350596,
        0.019515617366635,
        0.014865667824256,
        0.012848910366333,
        0.01367692747826,
        0.013168340447608,
        0.011705463134747,
        0.014168835136499,
        0.018696913763589,
        0.01122982868689,
        0.019563439481637,
        0.01518667984023,
        0.009402631911497,
        0.017219320577697,
        0.014078650283102,
        0.012928543934715,
        0.00947457479053,
        0.012432166798258,
        0.015457370217179,
        0.030879769701957,
        0.018252617818492,
        0.011240344124282,
        0.019895455602401,
        0.021611175750742,
        0.020503772897029,
        0.021298192972989,
        0.018962571011148,
        0.023028308496324,
        0.031770755867088,
        0.011911672152197,
        0.050818823024488,
        0.027288645057825,
        0.012965678635229,
        0.030722972905583,
        0.025265729520182,
        0.022137964432179,
        0.029347817041573,
        0.0233452842333,
        0.02481757350596,
        0.018252617818492,
        0.046731120991691,
        0.01787790548511,
        0.020958081509075,
        0.018065153819997,
        0.022205670397415,
        0.017434638452365,
        0.025201263005006,
        0.015492581331411,
        0.013480095835065,
        0.012304545525059,
        0.019635934227587,
        0.017631040262766,
        0.009099063413323,
        0.027274607535281,
        0.023889942860297,
        0.010120821228927,
        0.00884477964579,
        0.01926890291218,
        0.019515617366635,
        0.011240344124282,
        0.01787790548511,
        0.03994950179388,
    ],
    20,
    :,
)
bl.post_cov ≈ testpost_cov
testblweights = [
    -0.14033853984083458,
    -0.1179297240072182,
    -0.0678057021322054,
    -0.027246566593579404,
    -0.021522035742528572,
    -0.07847917478909809,
    -0.007664178545146779,
    0.01835946351868358,
    -0.09094473419577463,
    0.0015591227372739273,
    0.18021837301404844,
    0.0016670689495436144,
    0.0006889543533665808,
    0.16636619646552375,
    0.014691504346391951,
    0.044422733010109454,
    0.21766625222512298,
    0.3559421484619688,
    0.2128429410747911,
    0.3375058976895615,
]
bl.weights ≈ testblweights
mu, sigma, sr = portfolio_performance(bl)
mutest, sigmatest, srtest = 13.39180614873814, 0.14848981886018245, 90.05200660476926
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest

##########################################

bl = BlackLitterman(
    mcapsdf[!, 1],
    S;
    rf = 0,
    tau = 0.01,
    pi = prior, # either a vector, `nothing`, `:equal`, or `:market`
    Q = views,
    P = picking,
    absolute_views = Dict("GE" => 0.5, "SBUX" => -0.3, "BBY" => 1.3, "BAC" => -2.3),
)
testpost_ret = [
    -1.825463461532543e-01,
    -1.630999422010739e-01,
    -1.805491276247375e-01,
    -1.472669843163014e-01,
    -1.852452562514492e-01,
    4.683712686565655e-02,
    -3.352014678120480e-01,
    5.431551831423520e-04,
    -9.069592919799663e-01,
    -2.254724827015644e-01,
    -1.394661366884839e-02,
    -1.993725363741211e-01,
    -5.863851620670142e-02,
    -1.740509723864487e-01,
    -1.895919050736239e-01,
    4.127594574103763e-01,
    -2.404701101132104e-01,
    -1.416446824074105e-01,
    -5.982403219912473e-01,
    -1.657796851558853e-01,
]
bl.post_ret ≈ testpost_ret
testpost_cov = reshape(
    [
        0.053355004708403,
        0.024869372523866,
        0.035736393699335,
        0.03000673564508,
        0.040221179824619,
        0.015677827170267,
        0.020827985252809,
        0.010583538924501,
        0.022902060072788,
        0.018138721961276,
        0.010211125313537,
        0.025906912998953,
        0.008991361080725,
        0.013503850580997,
        0.006145859478046,
        0.012641179362347,
        0.024984890971051,
        0.014937824882293,
        0.019963981996037,
        0.021021498271408,
        0.024869372523866,
        0.05375606281688,
        0.027692809734518,
        0.023994557872118,
        0.027216630261856,
        0.014705308536151,
        0.033038695113238,
        0.011531878909644,
        0.02385399705785,
        0.019213528992292,
        0.010590034395602,
        0.023119692995778,
        0.01804745278545,
        0.014931458900159,
        0.013687324030089,
        0.018785247501608,
        0.02347490562648,
        0.012909641149555,
        0.02168795764157,
        0.018117268277973,
        0.035736393699335,
        0.027692809734518,
        0.065374073200538,
        0.030284300433481,
        0.038929394425814,
        0.016948377864133,
        0.029687734554031,
        0.009782627597552,
        0.023885547778898,
        0.019269431079962,
        0.009874629151569,
        0.03496589593972,
        0.014710786354096,
        0.013284834134868,
        0.010950170496824,
        0.014252763967257,
        0.025779327976905,
        0.013740657362995,
        0.02057300084152,
        0.022272256366145,
        0.03000673564508,
        0.023994557872118,
        0.030284300433481,
        0.101882271803604,
        0.033660267397293,
        0.016460774161477,
        0.050959910630609,
        0.008769740967924,
        0.024750610510677,
        0.020656398409812,
        0.007227146755567,
        0.028099971754321,
        0.025424559655084,
        0.014265920302375,
        0.024434638215355,
        0.019690452405807,
        0.025598228737059,
        0.013230091153188,
        0.021373340492103,
        0.017488700508079,
        0.040221179824619,
        0.027216630261856,
        0.038929394425814,
        0.033660267397293,
        0.084169295446172,
        0.013044654145067,
        0.030635075381508,
        0.009314816202311,
        0.022286370230474,
        0.016687824887789,
        0.009442855492669,
        0.030197754401657,
        0.004814001143552,
        0.01296878807647,
        0.019642613558522,
        0.014417526737778,
        0.026497967487756,
        0.011759687681249,
        0.019027965837789,
        0.025276079092208,
        0.015677827170267,
        0.014705308536151,
        0.016948377864133,
        0.016460774161477,
        0.013044654145067,
        0.045338367469502,
        0.023909779691727,
        0.00968022645508,
        0.02516308787993,
        0.021948589118261,
        0.013352267781383,
        0.026984265926953,
        0.029122907673343,
        0.017571128670328,
        0.025470962463853,
        0.012034563529256,
        0.016354149891514,
        0.014204076451461,
        0.023069602158711,
        0.015509542369282,
        0.020827985252809,
        0.033038695113238,
        0.029687734554031,
        0.050959910630609,
        0.030635075381508,
        0.023909779691727,
        0.419318902562546,
        0.011436682721276,
        0.042438354852465,
        0.035125063675608,
        0.016344247577001,
        0.02675654287593,
        0.035433366173133,
        0.022723909611437,
        0.061137074824638,
        0.030341104687899,
        0.033114161186038,
        0.018783848899739,
        0.031881531552557,
        0.013517880250322,
        0.010583538924501,
        0.011531878909644,
        0.009782627597552,
        0.008769740967924,
        0.009314816202311,
        0.00968022645508,
        0.011436682721276,
        0.041591573817594,
        0.010459791524178,
        0.012622860921192,
        0.010760733862937,
        0.016226711855118,
        0.028343706802912,
        0.00874523896063,
        0.005256498110719,
        0.01460599251905,
        0.011658585652992,
        0.011290480918447,
        0.01195743114572,
        0.012337824559677,
        0.022902060072788,
        0.02385399705785,
        0.023885547778898,
        0.024750610510677,
        0.022286370230474,
        0.02516308787993,
        0.042438354852465,
        0.010459791524178,
        0.069917017052211,
        0.031827326633377,
        0.011813491979717,
        0.035834635508991,
        0.03752391863838,
        0.023121776174192,
        0.034242095003063,
        0.029912584515676,
        0.027785044880916,
        0.01961903781816,
        0.050916005405932,
        0.019665678434845,
        0.018138721961276,
        0.019213528992292,
        0.019269431079962,
        0.020656398409812,
        0.016687824887789,
        0.021948589118261,
        0.035125063675608,
        0.012622860921192,
        0.031827326633377,
        0.061802541208702,
        0.012068829404024,
        0.032463351576325,
        0.036916659054851,
        0.016622005941096,
        0.026364903855968,
        0.025293265468425,
        0.019465649979454,
        0.015247986454704,
        0.027372783103943,
        0.017669559753497,
        0.010211125313537,
        0.010590034395602,
        0.009874629151569,
        0.007227146755567,
        0.009442855492669,
        0.013352267781383,
        0.016344247577001,
        0.010760733862937,
        0.011813491979717,
        0.012068829404024,
        0.026783192023305,
        0.012363120983916,
        0.01490648954212,
        0.012310365894262,
        0.015862641884786,
        0.010187197030165,
        0.010803880330312,
        0.009449647821345,
        0.013014720997322,
        0.009122220631128,
        0.025906912998953,
        0.023119692995778,
        0.03496589593972,
        0.028099971754321,
        0.030197754401657,
        0.026984265926953,
        0.02675654287593,
        0.016226711855118,
        0.035834635508991,
        0.032463351576325,
        0.012363120983916,
        0.179656620114785,
        0.05835947739786,
        0.019159213164975,
        0.033067889199988,
        0.034773247881559,
        0.023528678791207,
        0.017283986289006,
        0.030813450049317,
        0.027341083010291,
        0.008991361080725,
        0.01804745278545,
        0.014710786354096,
        0.025424559655084,
        0.004814001143552,
        0.029122907673343,
        0.035433366173133,
        0.028343706802912,
        0.03752391863838,
        0.036916659054851,
        0.01490648954212,
        0.05835947739786,
        0.494500990298495,
        0.019818179134617,
        0.056233497828945,
        0.059174653324722,
        0.015029629171534,
        0.014112869906447,
        0.025310564485113,
        0.023931001023522,
        0.013503850580997,
        0.014931458900159,
        0.013284834134868,
        0.014265920302375,
        0.01296878807647,
        0.017571128670328,
        0.022723909611437,
        0.00874523896063,
        0.023121776174192,
        0.016622005941096,
        0.012310365894262,
        0.019159213164975,
        0.019818179134617,
        0.036494762053532,
        0.039241479095165,
        0.011944914367486,
        0.016326670116902,
        0.012988799859125,
        0.022218724037443,
        0.010145498823839,
        0.006145859478046,
        0.013687324030089,
        0.010950170496824,
        0.024434638215355,
        0.019642613558522,
        0.025470962463853,
        0.061137074824638,
        0.005256498110719,
        0.034242095003063,
        0.026364903855968,
        0.015862641884786,
        0.033067889199988,
        0.056233497828945,
        0.039241479095165,
        0.25277417724088,
        0.028956898932882,
        0.016958839027437,
        0.009501034120303,
        0.02945197733489,
        0.008856849420103,
        0.012641179362347,
        0.018785247501608,
        0.014252763967257,
        0.019690452405807,
        0.014417526737778,
        0.012034563529256,
        0.030341104687899,
        0.01460599251905,
        0.029912584515676,
        0.025293265468425,
        0.010187197030165,
        0.034773247881559,
        0.059174653324722,
        0.011944914367486,
        0.028956898932882,
        0.127987590224161,
        0.018896646886567,
        0.012457154653794,
        0.023378182728825,
        0.019284813708051,
        0.024984890971051,
        0.02347490562648,
        0.025779327976905,
        0.025598228737059,
        0.026497967487756,
        0.016354149891514,
        0.033114161186038,
        0.011658585652992,
        0.027785044880916,
        0.019465649979454,
        0.010803880330312,
        0.023528678791207,
        0.015029629171534,
        0.016326670116902,
        0.016958839027437,
        0.018896646886567,
        0.041244172681506,
        0.01552826869779,
        0.024900624111878,
        0.019565761477073,
        0.014937824882293,
        0.012909641149555,
        0.013740657362995,
        0.013230091153188,
        0.011759687681249,
        0.014204076451461,
        0.018783848899739,
        0.011290480918447,
        0.01961903781816,
        0.015247986454704,
        0.009449647821345,
        0.017283986289006,
        0.014112869906447,
        0.012988799859125,
        0.009501034120303,
        0.012457154653794,
        0.01552826869779,
        0.031009299950532,
        0.01831707367678,
        0.011269360478198,
        0.019963981996037,
        0.02168795764157,
        0.02057300084152,
        0.021373340492103,
        0.019027965837789,
        0.023069602158711,
        0.031881531552557,
        0.01195743114572,
        0.050916005405932,
        0.027372783103943,
        0.013014720997322,
        0.030813450049317,
        0.025310564485113,
        0.022218724037443,
        0.02945197733489,
        0.023378182728825,
        0.024900624111878,
        0.01831707367678,
        0.046839850014613,
        0.017910031267031,
        0.021021498271408,
        0.018117268277973,
        0.022272256366145,
        0.017488700508079,
        0.025276079092208,
        0.015509542369282,
        0.013517880250322,
        0.012337824559677,
        0.019665678434845,
        0.017669559753497,
        0.009122220631128,
        0.027341083010291,
        0.023931001023522,
        0.010145498823839,
        0.008856849420103,
        0.019284813708051,
        0.019565761477073,
        0.011269360478198,
        0.017910031267031,
        0.039973688065651,
    ],
    20,
    :,
)
bl.post_cov ≈ testpost_cov
testblweights = [
    -0.1688512279401054,
    -0.23601888490258272,
    -0.11358961945612286,
    -0.11522237155854398,
    -0.1632599143596488,
    -4.354710542828925,
    -0.002057445643144843,
    -0.14354525492385362,
    8.534945058689114,
    -0.02004480102544189,
    -0.17683014345527887,
    -0.0016705812255951005,
    0.005339620990441324,
    -0.13835769795903335,
    -0.0006321689699330831,
    -3.005861249419085,
    -0.06032450794994511,
    -0.15716350632104556,
    -0.03837032297793442,
    1.356225561236661,
]
bl.weights ≈ testblweights
mu, sigma, sr = portfolio_performance(bl)
mutest, sigmatest, srtest = -9.18176341770092, 1.9905099079657798, -4.6228171891416245
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest