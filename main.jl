using Statistics, StatsBase, PortfolioOptimiser
using CSV, DataFrames, TimeSeries, LinearAlgebra, JuMP, Ipopt, ECOS

# Reading in the data; preparing expected returns and a risk model
df = CSV.read("./test/assets/stock_prices.csv", DataFrame)
tickers = names(df)[2:end]
returns = dropmissing(returns_from_prices(df[!, 2:end]))

mean_ret = ret_model(MRet(), Matrix(returns))
S = risk_matrix(Cov(), Matrix(returns))

ef = MeanVar(tickers, mean_ret, S)
sectors = ["Tech", "Medical", "RealEstate", "Oil"]
sector_map = Dict(tickers[1:4:20] .=> sectors[1])
merge!(sector_map, Dict(tickers[2:4:20] .=> sectors[2]))
merge!(sector_map, Dict(tickers[3:4:20] .=> sectors[3]))
merge!(sector_map, Dict(tickers[4:4:20] .=> sectors[4]))
sector_lower =
    Dict([(sectors[1], 0.2), (sectors[2], 0.1), (sectors[3], 0.15), (sectors[4], 0.05)])
sector_upper =
    Dict([(sectors[1], 0.4), (sectors[2], 0.5), (sectors[3], 0.2), (sectors[4], 0.2)])
add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)
add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)

max_sharpe!(ef)

0.2 - 1e-9 <= sum(ef.weights[1:4:20]) <= 0.4 + 1e-9
0.1 - 1e-9 <= sum(ef.weights[2:4:20]) <= 0.5 + 1e-9
0.15 - 1e-9 <= sum(ef.weights[3:4:20]) <= 0.2 + 1e-9
0.05 - 1e-9 <= sum(ef.weights[4:4:20]) <= 0.2 + 1e-9

ef = MeanSemivar(tickers, mean_ret, S, market_neutral = true)
add_sector_constraint!(ef, sector_map, sector_lower, sector_upper)

function sharpe_ratio_nl(w::T...) where {T}
    mean_ret = obj_params[1]
    cov_mtx = obj_params[2]
    rf = obj_params[3]

    w = [i for i in w]
    sr = PortfolioOptimiser.sharpe_ratio(w, mean_ret, cov_mtx, rf)

    return -sr
end
ef = MeanVar(tickers, mean_ret, S)
obj_params = [ef.mean_ret, ef.cov_mtx, ef.rf]
@time custom_nloptimiser!(ef, sharpe_ratio_nl, obj_params)
mu, sigma, sr = portfolio_performance(ef, verbose = true)

ef2 = MeanVar(tickers, mean_ret, S)
@time max_sharpe!(ef2)
mu2, sigma2, sr2 = portfolio_performance(ef2, verbose = true)

isapprox(mu, mu2, rtol = 1e-6)
isapprox(sigma, sigma2, rtol = 1e-6)
isapprox(sr, sr2, rtol = 1e-6)

cv = EfficientCVaR(tickers, mean_ret, Matrix(returns))
function cvar_ratio(w...)
    mean_ret = obj_params[1]
    beta = obj_params[2]
    samples = obj_params[3]
    n = obj_params[4]
    o = obj_params[5]
    p = obj_params[6]

    weights = [i for i in w[1:n]]
    alpha = w[o]
    u = [i for i in w[p:end]]

    mu = PortfolioOptimiser.port_return(weights, mean_ret)
    cv = PortfolioOptimiser.cdar(alpha, u, samples, beta)

    return -mu / cv
end

obj_params = []
extra_vars = []
push!(obj_params, mean_ret)
push!(obj_params, cv.beta)
push!(obj_params, size(cv.returns, 1))
push!(obj_params, 20)
push!(obj_params, 21)
push!(obj_params, 22)
push!(extra_vars, (cv.model[:alpha], 0.1))
push!(extra_vars, (cv.model[:u], fill(1 / length(cv.model[:u]), length(cv.model[:u]))))
custom_nloptimiser!(cv, cvar_ratio, obj_params, extra_vars)
mu, cvar = portfolio_performance(cv, verbose = true)
mu / cvar

cv2 = EfficientCVaR(tickers, mean_ret, Matrix(returns))
min_cvar!(cv2)
mu2, cvar2 = portfolio_performance(cv2, verbose = true)
mu2 / cvar2

cd = EfficientCDaR(tickers, mean_ret, Matrix(returns))
function cdar_ratio(w...)
    mean_ret = obj_params[1]
    beta = obj_params[2]
    samples = obj_params[3]
    n = obj_params[4]
    o = obj_params[5]
    p = obj_params[6]

    weights = [i for i in w[1:n]]
    alpha = w[o]
    z = [i for i in w[p:end]]

    mu = PortfolioOptimiser.port_return(weights, mean_ret)
    cd = PortfolioOptimiser.cdar(alpha, z, samples, beta)

    return -mu / cd
end

obj_params = []
extra_vars = []
push!(obj_params, mean_ret)
push!(obj_params, cd.beta)
push!(obj_params, size(cd.returns, 1))
push!(obj_params, 20)
push!(obj_params, 21)
push!(obj_params, 22)
push!(extra_vars, (cd.model[:alpha], 0.1))
push!(extra_vars, (cd.model[:z], fill(1 / length(cd.model[:z]), length(cd.model[:z]))))
custom_nloptimiser!(cd, cdar_ratio, obj_params, extra_vars)
mu, cdar = portfolio_performance(cd, verbose = true)
mu / cdar ≈ 3.2819464291074305

cd2 = EfficientCDaR(tickers, mean_ret, Matrix(returns))
min_cdar!(cd2)
mu2, cdar2 = portfolio_performance(cd2, verbose = true)
mu2 / cdar2

mu / cdar > mu2 / cdar2

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
lower_bounds, upper_bounds, strict_bounds = 0.02, 0.03, 0.1
ef = MeanVar(
    tickers,
    mu,
    S;
    extra_constraints = [
        :(model[:w][1] >= $lower_bounds),
        :(model[:w][end] <= $upper_bounds),
        :(model[:w][6] == $strict_bounds),
    ],
)
obj_params = [ef.mean_ret, ef.cov_mtx, 100]
custom_optimiser!(ef, kelly_objective, obj_params; silent = false)
ef.weights[1] >= lower_bounds
ef.weights[end] <= upper_bounds
isapprox(ef.weights[6], strict_bounds)

portfolio_performance(ef, verbose = true)

kelly_objective(fill(1 / 20, 20), obj_params[1], obj_params[2], obj_params[3])

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

mean_ret = ret_model(MRet(), Matrix(returns))
ef = MeanVar(tickers, mean_ret, S)
max_sharpe!(ef)
sr1 = sharpe_ratio(ef.weights, ef.mean_ret, ef.cov_mtx)
mu, sigma, sr = portfolio_performance(ef)
sr ≈ sr1
L2_reg(ef.weights, 0.69) ≈ 0.2823863907490376
transaction_cost(ef.weights, fill(1 / 20, 20), 0.005) ≈ 0.007928708353566113
ex_ante_tracking_error(ef.weights, S, fill(1 / 20, 20)) ≈ 0.022384515086395627
ex_post_tracking_error(ef.weights, Matrix(returns), fill(1 / 895, 895)) ≈
0.16061468446601332

test = Matrix(returns) * ef.weights - fill(1 / 895, 895)
var(test, mean = 0)

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

### Market neutral

using PortfolioOptimiser, DataFrames, CSV, Statistics

df = CSV.read("./test/assets/stock_prices.csv", DataFrame)
dropmissing!(df)
returns = returns_from_prices(df[!, 2:end])
tickers = names(df)[2:end]

mu = vec(ret_model(MRet(), Matrix(returns)))
S = risk_matrix(Cov(), Matrix(returns))

hrp = HRPOpt(tickers, returns = Matrix(returns))
optimise!(hrp, max_quadratic_utility!)
testweights = [
    0.05322978571963428,
    0.0013525614095372441,
    0.0845701131689161,
    0.08166673007984035,
    0.16404173578207706,
    0.0006158870845536016,
    0.049189141600734725,
    0.0069323722289936765,
    0.05932923127907018,
    0.061314572056272124,
    0.023130441823266446,
    0.005234336476474105,
    0.029075109239338887,
    0.009122881135802348,
    0.08064447242628374,
    0.1282639687104863,
    0.16244508468678345,
    0.03794588221718694,
    0.06687051853947568,
    0.06890700229846011,
]
hrp.weights ≈ testweights
mu, sigma, sr = portfolio_performance(hrp)
mutest, sigmatest, srtest = 0.23977731989029213, 0.1990931650935869, 1.103891837708152
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest

hrp = HRPOpt(tickers, returns = Matrix(returns))
optimise!(hrp, max_return!)
testweights = [
    0.053215775973689344,
    0.0014298867467719253,
    0.08459870820429168,
    0.08164522293333767,
    0.16394762797243273,
    0.0006506856473305975,
    0.04877541431712327,
    0.006966529106643553,
    0.05933305965116203,
    0.061351686897633866,
    0.023122626771250834,
    0.005255850404371056,
    0.028772641690166374,
    0.009101244131990852,
    0.08058316841784602,
    0.1283171317054279,
    0.1623978785944996,
    0.03793881981412327,
    0.06685508940506842,
    0.0688799562236672,
]
hrp.weights ≈ testweights
mu, sigma, sr = portfolio_performance(hrp)
mutest, sigmatest, srtest = 0.23967354225156112, 0.19889330638531708, 1.1044793122699987
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest

optimise!(hrp, max_quadratic_utility!, 1e-12)
mu_mq, sigma_mq, sr_mq = portfolio_performance(hrp)
mu ≈ mu_mq
sigma ≈ sigma_mq
sr ≈ sr_mq

optimise!(hrp, min_volatility!)
mu, sigma, sr = portfolio_performance(hrp)
optimise!(hrp, max_quadratic_utility!, 1e12)
mu_mq, sigma_mq, sr_mq = portfolio_performance(hrp)
mu ≈ mu_mq
sigma ≈ sigma_mq
sr ≈ sr_mq

hrp = HRPOpt(tickers, returns = Matrix(returns))
optimise!(hrp, max_sharpe!)
testweights = [
    0.060950709858938946,
    0.024688958223226046,
    0.07636491146846057,
    0.09098114189215895,
    0.16345933040156524,
    0.01321315607100166,
    0.030495530737540685,
    0.0041676003148035014,
    0.05012315893824599,
    0.047425984889118235,
    0.0354866498145466,
    0.0029681045872571015,
    0.018207753673800363,
    0.01757206461257125,
    0.07232501039144949,
    0.09318121978905623,
    0.17882024226595147,
    0.04731099202207161,
    0.0699409077092408,
    0.0859547509506454,
]
hrp.weights ≈ testweights
mu, sigma, sr = portfolio_performance(hrp)
mutest, sigmatest, srtest = 0.23690508565115886, 0.19266051071806212, 1.1258409148960271
mu ≈ mutest
sigma ≈ sigmatest
sr ≈ srtest

optimise!(hrp, min_volatility!)
mu1, sigma1, sr1 = portfolio_performance(hrp)
optimise!(hrp, max_return!)
mu2, sigma2, sr2 = portfolio_performance(hrp)
optimise!(hrp, max_quadratic_utility!)
mu3, sigma3, sr3 = portfolio_performance(hrp)

sr > sr1
sr > sr2
sr > sr3
