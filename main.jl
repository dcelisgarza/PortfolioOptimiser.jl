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

## Mean port_semivar

### Mean port_semivar market neutral

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

ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns))

ef = MeanSemivar(
    tickers,
    bl.post_ret,
    Matrix(returns),
    weight_bounds = (0, 1),
    market_neutral = false,
)
max_sortino!(ef)
mumax, varmax, smax = portfolio_performance(ef)

ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns), weight_bounds = (0, 1))

ef = MeanSemivar(tickers, bl.post_ret, Matrix(returns), market_neutral = true)
