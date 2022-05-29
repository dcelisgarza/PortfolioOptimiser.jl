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

### Eff front market neutral
ef = MeanVar(
    names(df)[2:end],
    bl.post_ret,
    S,
    market_neutral = true,
    weight_bounds = (-1, 1),
)

## CVaR

### Market neutral

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
tickers = names(df)[2:end]

mu = vec(ret_model(MRet(), Matrix(returns)))
S = risk_matrix(Cov(), Matrix(returns))

spy_prices = CSV.read("./test/assets/spy_prices.csv", DataFrame)
delta = market_implied_risk_aversion(spy_prices[!, 2])
# In the order of the dataframes, the
mcapsdf = DataFrame(
    ticker = tickers,
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

cv =
    EfficientCVaR(tickers, bl.post_ret, Matrix(returns), beta = 0.2, market_neutral = false)
min_cvar!(cv)
testweights = [
    1.2591e-12,
    0.0239999111487575,
    0.0650098845478511,
    0.0531407521736819,
    0.0990996575024338,
    3.624e-13,
    0.0106099759402764,
    0.1031880979043503,
    8.525e-13,
    0.0238096685816571,
    0.2741982381970403,
    8.64e-14,
    6.49e-14,
    0.0021107467292905,
    1.032e-13,
    0.0302164465597255,
    0.053838138362372,
    0.1363179609151427,
    0.0210516374364299,
    0.1034088839982625,
]
isapprox(cv.weights, testweights, rtol = 1e-2)
mu, cvar = portfolio_performance(cv)
mutest, cvartest = 0.015141313656655424, 0.0020571077411184347
isapprox(mu, mutest, rtol = 1e-2)
isapprox(cvar, cvartest, rtol = 1e-2)

efficient_return!(cv, 0.09)
testweights = [
    5.398e-13,
    1.53e-14,
    2.9e-15,
    0.0208931899503421,
    2.98e-14,
    1.4e-15,
    0.0561037656427129,
    3.6e-15,
    0.9230030444062806,
    4.5e-15,
    2.6e-15,
    1.9e-15,
    7e-16,
    7e-15,
    3.9e-15,
    4.1e-15,
    8e-15,
    6.5e-15,
    3.2e-14,
    2e-16,
]
isapprox(cv.weights, testweights, rtol = 1e-3)
mu, cvar = portfolio_performance(cv)
mutest, cvartest = 0.09000000000000004, 0.004545680961201856
isapprox(mu, mutest, rtol = 1e-6)
isapprox(cvar, cvartest, rtol = 1e-3)

efficient_risk!(cv, 0.1438)
testweights = [
    2.1e-14,
    1.09e-14,
    -2.19e-14,
    5.82e-14,
    -1.67e-14,
    -1.52e-14,
    4.488e-12,
    -3.38e-14,
    0.9999999999954172,
    1.4e-15,
    -3.02e-14,
    -4.3e-15,
    -1.04e-14,
    6.9e-15,
    9.49e-14,
    -1.49e-14,
    -5.5e-15,
    -1.65e-14,
    8.64e-14,
    -1.56e-14,
]
isapprox(cv.weights, testweights, rtol = 1e-6)
mu, cvar = portfolio_performance(cv)
mutest, cvartest = 0.09141921970332006, 0.14377337204585958
isapprox(mu, mutest, rtol = 1e-6)
isapprox(cvar, cvartest, rtol = 1e-3)